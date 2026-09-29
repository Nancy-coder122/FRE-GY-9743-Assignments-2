from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import QuantLib as ql

from fixedincomelib.apis.date import qfCreateSchedule
from fixedincomelib.date import Date, Period, accrued, add_period
from fixedincomelib.market.basics import AccrualBasis, BusinessDayConvention, HolidayConvention


__all__ = ["BondCalculator"]


class BondCalculator:
    """Fixed-rate bullet bond built from user-supplied terms, without case IDs."""
    def __init__(self, issue_date: Optional[str] = None,
                 first_acc_date: Optional[str] = None,
                 first_cpn_date: Optional[str] = None,
                 maturity_date: Optional[str] = None,
                 coupon_rate: Optional[float] = None,
                 bond_convention: Optional[Dict[str, Any]] = None,
                 face_value: Optional[float] = None,
                 redemption: Optional[float] = None) -> None:
        self.conv = dict(bond_convention)
        self.issue_date = issue_date if issue_date is not None else self.conv['issue date']
        self.first_acc_date = first_acc_date if first_acc_date is not None else self.conv['first accrual date']
        self.first_cpn_date = first_cpn_date if first_cpn_date is not None else self.conv['first coupon date']
        self.maturity_date = maturity_date if maturity_date is not None else self.conv['maturity date']
        self.coupon_rate = coupon_rate if coupon_rate is not None else self.conv['coupon rate']
        self.face_value = face_value if face_value is not None else self.conv['face value']
        self.redemption = redemption if redemption is not None else self.conv['redemption']
        self.coupon_period = Period(self.conv['coupon accrual period'])
        self.frequency = float(self.coupon_period.frequency())
        self.accrual_basis = AccrualBasis(self.conv['accrual basis'])
        self.first_accrual_basis = AccrualBasis(self.conv['first period accrual basis'])
        self.last_accrual_basis = AccrualBasis(self.conv['last period accrual basis'])
        self.schedule = self._generate_bond_schedule()
        for period in self.schedule:
            if period['period_type'] == 'first':
                period['coupon'] = self._first_period_accrual(period)
            elif period['period_type'] == 'last':
                period['coupon'] = self._last_period_accrual(period)
            else:
                period['coupon'] = self._regular_period_accrual(period)

    def settlement_date(self, value_date: str) -> Date:
        return add_period(
            Date(value_date), Period(self.conv['settlement offset']),
            BusinessDayConvention.new(self.conv['settlement business day convention']),
            HolidayConvention.new(self.conv['settlement holiday convention']))

    def yield_to_price(self, value_date: Optional[str] = None,
                       yield_rate: Optional[float] = None, *,
                       settlement_date: Optional[str] = None) -> Dict[str, Any]:
        settle = (Date(settlement_date) if settlement_date is not None
                  else self.settlement_date(value_date))
        remaining = [period for period in self.schedule
                     if period['end_date'] > settle]
        if not remaining:
            raise ValueError('settlement date must be before maturity')

        coupon_amounts = np.asarray(
            [period['coupon'] for period in remaining], dtype=float)
        increments = [self._periods_to_next_coupon(settle, remaining[0])]
        increments.extend(
            self._periods_to_next_coupon(period['start_date'], period)
            for period in remaining[1:]
        )
        n = np.cumsum(np.asarray(increments, dtype=float))

        y = float(yield_rate)
        f = self.frequency
        if len(remaining) == 1:
            t = n / f
            denominator = 1.0 + y * t
            discount = 1.0 / denominator
            discount_first = -t / denominator**2
            discount_second = 2.0 * t**2 / denominator**3
        else:
            base = 1.0 + y / f
            discount = base**(-n)
            discount_first = -n / f * base**(-n - 1.0)
            discount_second = n * (n + 1.0) / f**2 * base**(-n - 2.0)

        redemption_amount = self.face_value * self.redemption
        dirty = (np.dot(coupon_amounts, discount)
                 + redemption_amount * discount[-1])
        ai = self._accrued_interest(settle)
        first = (np.dot(coupon_amounts, discount_first)
                 + redemption_amount * discount_first[-1])
        second = (np.dot(coupon_amounts, discount_second)
                  + redemption_amount * discount_second[-1])
        return {'clean_price': float(dirty - ai), 'dirty_price': float(dirty),
                'accrued_interest': ai, 'dpricedyield': float(first),
                'd2pricedyield2': float(second), 'settlement_date': settle.ISO()}

    def _generate_bond_schedule(self) -> List[Dict[str, Any]]:
        first_regular_date = (
            None if self.first_cpn_date == self.maturity_date
            else self.first_cpn_date
        )
        next_to_last_date = self.conv['last regular coupon date']
        end_of_month = self.conv['end of month']

        schedule_frame = qfCreateSchedule(
            start_date=self.first_acc_date,
            end_date=self.maturity_date,
            accrual_period=self.conv['coupon accrual period'],
            holiday_convention='NONE',
            business_day_convention='NONE',
            accrual_basis='NONE',
            rule=self.conv['schedule rule'],
            end_of_month=end_of_month,
            payment_offset=self.conv['payment offset'],
            payment_offset_business_day_convention=(
                self.conv['payment business day convention']),
            payment_offset_holiday_convention=(
                self.conv['payment holiday convention']),
            first_regular_date=first_regular_date,
            next_to_last_date=next_to_last_date,
        )

        rule = (ql.DateGeneration.Forward
                if self.conv['schedule rule'].upper() == 'FORWARD'
                else ql.DateGeneration.Backward)
        native_schedule = ql.Schedule(
            Date(self.first_acc_date),
            Date(self.maturity_date),
            self.coupon_period,
            ql.NullCalendar(),
            ql.Unadjusted,
            ql.Unadjusted,
            rule,
            end_of_month,
            Date(first_regular_date),
            Date(next_to_last_date),
        )

        periods = []
        last_index = len(schedule_frame) - 1
        for i, row in schedule_frame.iterrows():
            start_date = Date(row['StartDate'])
            end_date = Date(row['EndDate'])
            expected_end = add_period(
                start_date, self.coupon_period,
                BusinessDayConvention.new('NONE'),
                HolidayConvention.new('NONE'), end_of_month)
            is_regular = bool(native_schedule.isRegular(i + 1)) or expected_end == end_date

            if i == 0:
                period_type = 'first'
                accrual_basis = self.first_accrual_basis
            elif i == last_index:
                period_type = 'last'
                accrual_basis = self.last_accrual_basis
            else:
                period_type = 'regular'
                accrual_basis = self.accrual_basis

            periods.append({
                'start_date': start_date,
                'end_date': end_date,
                'payment_date': Date(row['PaymentDate']),
                'is_regular': is_regular,
                'period_type': period_type,
                'accrual_basis': accrual_basis,
            })
        return periods

    def _first_period_accrual(self, period: Dict[str, Any]) -> float:
        if period['is_regular']:
            return self._regular_period_accrual(period)

        basis = period['accrual_basis']
        start_date = period['start_date']
        end_date = period['end_date']
        ref_end = end_date
        ref_start = add_period(
            ref_end, Period.negate_period(self.coupon_period),
            BusinessDayConvention.new('NONE'),
            HolidayConvention.new('NONE'), self.conv['end of month'])
        period['ref_start'] = ref_start
        period['ref_end'] = ref_end

        if basis.needs_reference_period:
            full_accrual = 0.0
            grid_start, grid_end = ref_start, ref_end
            while grid_end > start_date:
                overlap_start = max(start_date, grid_start)
                full_accrual += self._year_fraction(
                    overlap_start, grid_end, basis, grid_start, grid_end)
                grid_end = grid_start
                grid_start = add_period(
                    grid_end, Period.negate_period(self.coupon_period),
                    BusinessDayConvention.new('NONE'),
                    HolidayConvention.new('NONE'), self.conv['end of month'])
        else:
            full_accrual = self._year_fraction(start_date, end_date, basis)

        period['accrual'] = full_accrual
        return self.face_value * self.coupon_rate * full_accrual

    def _last_period_accrual(self, period: Dict[str, Any]) -> float:
        if period['is_regular']:
            return self._regular_period_accrual(period)

        basis = period['accrual_basis']
        start_date = period['start_date']
        end_date = period['end_date']
        ref_start = start_date
        ref_end = add_period(
            ref_start, self.coupon_period,
            BusinessDayConvention.new('NONE'),
            HolidayConvention.new('NONE'), self.conv['end of month'])
        period['ref_start'] = ref_start
        period['ref_end'] = ref_end

        if basis.needs_reference_period:
            full_accrual = 0.0
            grid_start, grid_end = ref_start, ref_end
            while grid_start < end_date:
                overlap_end = min(end_date, grid_end)
                full_accrual += self._year_fraction(
                    grid_start, overlap_end, basis, grid_start, grid_end)
                grid_start = grid_end
                grid_end = add_period(
                    grid_start, self.coupon_period,
                    BusinessDayConvention.new('NONE'),
                    HolidayConvention.new('NONE'), self.conv['end of month'])
        else:
            full_accrual = self._year_fraction(start_date, end_date, basis)

        period['accrual'] = full_accrual
        return self.face_value * self.coupon_rate * full_accrual

    def _regular_period_accrual(self, period: Dict[str, Any]) -> float:
        period['ref_start'] = period['start_date']
        period['ref_end'] = period['end_date']
        period['accrual'] = self._year_fraction(
            period['start_date'], period['end_date'],
            period['accrual_basis'], period['ref_start'], period['ref_end'])
        return self.face_value * self.coupon_rate / self.frequency

    def _accrued_interest(self, settlement_date: Date) -> float:
        period = next(
            (item for item in self.schedule
             if item['end_date'] > settlement_date),
            None)
        if period is None or settlement_date <= period['start_date']:
            return 0.0

        ai_t = self._year_fraction(
            period['start_date'], settlement_date,
            period['accrual_basis'], period['ref_start'], period['ref_end'])
        full_period_length = period['accrual']
        return period['coupon'] * ai_t / full_period_length

    def price_to_yield(self, value_date: Optional[str] = None, price: Optional[float] = None,
                       clean: bool = True, max_iter: int = 100, tol: float = 1e-12, *,
                       settlement_date: Optional[str] = None) -> Dict[str, Any]:
        settle = Date(settlement_date) if settlement_date is not None else self.settlement_date(value_date)
        accrued_interest = self._accrued_interest(settle)
        dirty_price = price + accrued_interest if clean else price
        lower = self._zcb_yield_guess(settle, dirty_price)
        upper = max(self.coupon_rate, lower + 0.05)
        while self.yield_to_price(settlement_date=settle.ISO(), yield_rate=upper)['dirty_price'] > dirty_price:
            upper = 2 * upper + self.frequency

        for _ in range(max_iter):
            ytm = (lower + upper) / 2
            result = self.yield_to_price(settlement_date=settle.ISO(), yield_rate=ytm)
            residual = result['dirty_price'] - dirty_price
            if abs(residual) <= tol:
                break
            if residual > 0:
                lower = ytm
            else:
                upper = ytm
        return {'yield': float(ytm), **result}

    def _zcb_yield_guess(self, settlement_date: Date, dirty_price: float) -> float:
        remaining = [p for p in self.schedule if p['end_date'] > settlement_date]
        n = self._periods_to_next_coupon(settlement_date, remaining[0])
        for period in remaining[1:]:
            n += self._periods_to_next_coupon(period['start_date'], period)
        redemption_amount = self.face_value * self.redemption
        if len(remaining) == 1:
            return self.frequency * (redemption_amount / dirty_price - 1) / n
        return self.frequency * ((redemption_amount / dirty_price)**(1 / n) - 1)

    def schedule_dataframe(self) -> pd.DataFrame:
        rows = []
        for i, p in enumerate(self.schedule):
            principal = self.face_value * self.redemption if i == len(self.schedule) - 1 else 0.
            rows.append({'StartDate': p['start_date'].ISO(), 'EndDate': p['end_date'].ISO(),
                         'PaymentDate': p['payment_date'].ISO(),
                         'ReferenceStart': p['ref_start'].ISO(), 'ReferenceEnd': p['ref_end'].ISO(),
                         'IsRegular': p['is_regular'], 'Accrued': p['accrual'],
                         'Coupon': p['coupon'], 'Principal': principal, 'Total': p['coupon'] + principal})
        return pd.DataFrame(rows)

    def _periods_to_next_coupon(self, settlement_date, period):
        basis = period['accrual_basis']
        remaining = self._year_fraction(settlement_date, period['end_date'], basis,
                                        period['ref_start'], period['ref_end'])
        reference = self._year_fraction(period['ref_start'], period['ref_end'], basis,
                                        period['ref_start'], period['ref_end'])
        return remaining / reference

    def _year_fraction(self, start_date, end_date, basis, reference_period_start=None, reference_period_end=None):
        if basis.needs_reference_period:
            return basis.value.yearFraction(
                start_date, end_date, reference_period_start, reference_period_end)
        return accrued(start_date, end_date, basis.value,
                       BusinessDayConvention.new('NONE'), HolidayConvention.new('NONE'))
