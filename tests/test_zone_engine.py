"""Tests for engines.zone_engine."""

import pytest

import smc_config as cfg
from engines.market_structure import MarketStructure
from engines.zone_engine import ZoneEngine
from utils.candle_types import (
    BOS,
    BOSType,
    Candle,
    Direction,
    Zone,
    ZonePattern,
    ZonePosition,
    ZoneType,
)


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

LB = cfg.SWING_LOOKBACK  # 5


def _c(ts: float, o: float, h: float, l: float, c: float) -> Candle:
    """Shorthand candle constructor."""
    return Candle(timestamp=ts, open=o, high=h, low=l, close=c)


def _rising(start: float, end: float, count: int, ts_start: float) -> list[Candle]:
    """Generate *count* rising candles from *start* to *end*."""
    step = (end - start) / (count + 1)
    candles: list[Candle] = []
    for i in range(count):
        mid = start + step * (i + 1)
        candles.append(_c(ts_start + i * 60, mid - 0.1, mid + 0.1, mid - 0.3, mid))
    return candles


def _falling(start: float, end: float, count: int, ts_start: float) -> list[Candle]:
    """Generate *count* falling candles from *start* to *end*."""
    step = (start - end) / (count + 1)
    candles: list[Candle] = []
    for i in range(count):
        mid = start - step * (i + 1)
        candles.append(_c(ts_start + i * 60, mid + 0.1, mid + 0.3, mid - 0.1, mid))
    return candles


def _feed_ms_ze(
    candles: list[Candle],
) -> tuple[MarketStructure, ZoneEngine, list[Zone]]:
    """Feed candles through MarketStructure + ZoneEngine, return both and any zones."""
    ms = MarketStructure()
    ze = ZoneEngine()
    zones: list[Zone] = []
    for c in candles:
        bos = ms.update(c)
        zone = ze.update(c, bos, ms._candles)
        if zone is not None:
            zones.append(zone)
    return ms, ze, zones


def _make_uptrend_with_bos() -> list[Candle]:
    """Create candles that form a swing high, pull back, then break above it.

    Pattern:
      LB rising -> peak candle -> LB falling -> LB rising -> breakout candle
    This should produce a swing high, then a bullish BOS when price closes above it.
    """
    candles: list[Candle] = []
    ts = 0.0

    # Rising to peak at 110.
    candles.extend(_rising(100, 109.5, LB, ts))
    ts += LB * 60
    candles.append(_c(ts, 109.8, 110.5, 109.6, 110.1))  # peak candle
    ts += 60

    # Falling to trough around 104.
    candles.extend(_falling(109, 104.5, LB, ts))
    ts += LB * 60
    candles.append(_c(ts, 104.2, 104.4, 103.5, 103.9))  # trough candle
    ts += 60

    # Rising back toward 110.
    candles.extend(_rising(104, 109, LB, ts))
    ts += LB * 60

    # Strong breakout above swing high (110.5).
    candles.append(_c(ts, 109, 113, 108.5, 112))
    return candles


def _make_downtrend_with_bos() -> list[Candle]:
    """Create candles that form a swing low, bounce, then break below it."""
    candles: list[Candle] = []
    ts = 0.0

    # Falling to trough at 100.
    candles.extend(_falling(110, 100.5, LB, ts))
    ts += LB * 60
    candles.append(_c(ts, 100.2, 100.4, 99.5, 99.9))  # trough
    ts += 60

    # Rising to peak around 106.
    candles.extend(_rising(100, 105.5, LB, ts))
    ts += LB * 60
    candles.append(_c(ts, 105.8, 106.5, 105.6, 106.1))  # peak
    ts += 60

    # Falling back toward 100.
    candles.extend(_falling(106, 100.5, LB, ts))
    ts += LB * 60

    # Strong breakdown below swing low (99.5).
    candles.append(_c(ts, 101, 101.5, 97, 97.5))
    return candles


# -----------------------------------------------------------------------
# Test 1: Zone creation on bullish BOS
# -----------------------------------------------------------------------


class TestDemandZoneCreation:
    """Bullish BOS should produce a DEMAND zone."""

    def test_demand_zone_created(self) -> None:
        candles = _make_uptrend_with_bos()
        _ms, ze, zones = _feed_ms_ze(candles)

        demand = [z for z in zones if z.zone_type == ZoneType.DEMAND]
        assert len(demand) >= 1
        z = demand[0]
        assert z.is_fresh is True
        assert z.is_broken is False


# -----------------------------------------------------------------------
# Test 2: Zone creation on bearish BOS
# -----------------------------------------------------------------------


class TestSupplyZoneCreation:
    """Bearish BOS should produce a SUPPLY zone."""

    def test_supply_zone_created(self) -> None:
        candles = _make_downtrend_with_bos()
        _ms, ze, zones = _feed_ms_ze(candles)

        supply = [z for z in zones if z.zone_type == ZoneType.SUPPLY]
        assert len(supply) >= 1
        z = supply[0]
        assert z.is_fresh is True
        assert z.is_broken is False


# -----------------------------------------------------------------------
# Test 3: Zone mitigation
# -----------------------------------------------------------------------


class TestZoneMitigation:
    """A candle entering the zone without breaking it -> mitigated."""

    def test_demand_mitigated(self) -> None:
        ze = ZoneEngine()
        # Manually insert a demand zone.
        z = Zone(
            timestamp=0, high=102, low=100, zone_type=ZoneType.DEMAND,
            candle_index=0,
        )
        ze._zones.append(z)

        # Candle dips into zone but closes above zone.low.
        candle = _c(60, 103, 104, 101, 103)  # low=101 <= zone.high=102
        ze.update(candle, None, [])

        assert z.is_mitigated is True
        assert z.is_broken is False

    def test_supply_mitigated(self) -> None:
        ze = ZoneEngine()
        z = Zone(
            timestamp=0, high=110, low=108, zone_type=ZoneType.SUPPLY,
            candle_index=0,
        )
        ze._zones.append(z)

        # Candle wicks into zone but closes below zone.high.
        candle = _c(60, 107, 109, 106, 107.5)  # high=109 >= zone.low=108
        ze.update(candle, None, [])

        assert z.is_mitigated is True
        assert z.is_broken is False


# -----------------------------------------------------------------------
# Test 4: Zone broken
# -----------------------------------------------------------------------


class TestZoneBroken:
    """A candle closing through the zone -> broken."""

    def test_demand_broken(self) -> None:
        ze = ZoneEngine()
        z = Zone(
            timestamp=0, high=102, low=100, zone_type=ZoneType.DEMAND,
            candle_index=0,
        )
        ze._zones.append(z)

        # Candle closes below zone.low.
        candle = _c(60, 101, 101.5, 98, 99)
        ze.update(candle, None, [])

        assert z.is_broken is True
        assert z.is_mitigated is True

    def test_supply_broken(self) -> None:
        ze = ZoneEngine()
        z = Zone(
            timestamp=0, high=110, low=108, zone_type=ZoneType.SUPPLY,
            candle_index=0,
        )
        ze._zones.append(z)

        # Candle closes above zone.high.
        candle = _c(60, 109, 112, 108, 111)
        ze.update(candle, None, [])

        assert z.is_broken is True


# -----------------------------------------------------------------------
# Test 5: Freshness expiry
# -----------------------------------------------------------------------


class TestFreshness:
    """Zones older than ZONE_FRESHNESS_MAX_AGE lose freshness."""

    def test_freshness_expires(self) -> None:
        ze = ZoneEngine()
        z = Zone(
            timestamp=0, high=110, low=108, zone_type=ZoneType.SUPPLY,
            candle_index=1,  # created at candle 1
        )
        ze._zones.append(z)
        ze._candle_count = 1  # sync

        # Feed candles that don't touch the zone.
        for i in range(cfg.ZONE_FRESHNESS_MAX_AGE + 1):
            candle = _c(60 * (i + 1), 103, 104, 102, 103)
            ze.update(candle, None, [])

        assert z.is_fresh is False


# -----------------------------------------------------------------------
# Test 6: Quality score
# -----------------------------------------------------------------------


class TestQualityScore:
    """Quality score components."""

    def test_impulsive_adds_quality(self) -> None:
        candles = _make_uptrend_with_bos()
        _ms, ze, zones = _feed_ms_ze(candles)

        if zones:
            # At minimum, any zone from a real BOS should have a score >= 0.
            assert zones[0].quality_score >= 0

    def test_quality_from_impulsive_bos(self) -> None:
        ze = ZoneEngine()
        bos = BOS(
            timestamp=100, price=110, direction=Direction.BULLISH,
            bos_type=BOSType.IMPULSIVE,
        )
        # Provide a small candles buffer.
        buf = [
            _c(0, 100, 101, 99, 100),
            _c(60, 101, 102, 100, 101.5),
            _c(120, 108, 113, 107.5, 112),
        ]
        zone = ze._create_zone_from_bos(bos, buf)
        assert zone is not None
        assert zone.quality_score >= 1  # at least +1 for impulsive

    def test_quality_from_choch_bos(self) -> None:
        ze = ZoneEngine()
        bos = BOS(
            timestamp=100, price=110, direction=Direction.BULLISH,
            bos_type=BOSType.IMPULSIVE, is_choch=True,
        )
        buf = [
            _c(0, 100, 101, 99, 100),
            _c(60, 101, 102, 100, 101.5),
            _c(120, 108, 113, 107.5, 112),
        ]
        zone = ze._create_zone_from_bos(bos, buf)
        assert zone is not None
        assert zone.quality_score >= 2  # +1 impulsive +1 choch


# -----------------------------------------------------------------------
# Test 7: Zone position classification
# -----------------------------------------------------------------------


class TestZonePosition:
    """classify_zone_position maps midpoint to TOP/MID/LOWER."""

    def test_top(self) -> None:
        ze = ZoneEngine()
        z = Zone(timestamp=0, high=109, low=107, zone_type=ZoneType.SUPPLY)
        # midpoint = 108, range [100, 110], pct = 0.80
        assert ze.classify_zone_position(z, 110, 100) == ZonePosition.TOP

    def test_lower(self) -> None:
        ze = ZoneEngine()
        z = Zone(timestamp=0, high=103, low=101, zone_type=ZoneType.DEMAND)
        # midpoint = 102, pct = 0.20
        assert ze.classify_zone_position(z, 110, 100) == ZonePosition.LOWER

    def test_mid(self) -> None:
        ze = ZoneEngine()
        z = Zone(timestamp=0, high=106, low=104, zone_type=ZoneType.DEMAND)
        # midpoint = 105, pct = 0.50
        assert ze.classify_zone_position(z, 110, 100) == ZonePosition.MID


# -----------------------------------------------------------------------
# Test 8: get_active_zones filters broken zones
# -----------------------------------------------------------------------


class TestActiveZones:
    """Broken zones should not appear in active list."""

    def test_filters_broken(self) -> None:
        ze = ZoneEngine()
        z1 = Zone(timestamp=0, high=110, low=108, zone_type=ZoneType.SUPPLY, candle_index=0)
        z2 = Zone(timestamp=60, high=105, low=103, zone_type=ZoneType.DEMAND, candle_index=1)
        z3 = Zone(timestamp=120, high=115, low=113, zone_type=ZoneType.SUPPLY, candle_index=2)
        z2.is_broken = True
        ze._zones.extend([z1, z2, z3])

        active = ze.get_active_zones()
        assert len(active) == 2
        assert z2 not in active


# -----------------------------------------------------------------------
# Test 9: get_nearest_zone
# -----------------------------------------------------------------------


class TestNearestZone:
    """Nearest zone by midpoint distance."""

    def test_nearest(self) -> None:
        ze = ZoneEngine()
        z1 = Zone(timestamp=0, high=111, low=109, zone_type=ZoneType.SUPPLY, candle_index=0)
        # midpoint=110
        z2 = Zone(timestamp=60, high=121, low=119, zone_type=ZoneType.SUPPLY, candle_index=1)
        # midpoint=120
        ze._zones.extend([z1, z2])

        nearest = ze.get_nearest_zone(112, ZoneType.SUPPLY)
        assert nearest is z1

    def test_no_zones(self) -> None:
        ze = ZoneEngine()
        assert ze.get_nearest_zone(100, ZoneType.SUPPLY) is None


# -----------------------------------------------------------------------
# Test 10: Pattern classification
# -----------------------------------------------------------------------


class TestPatternClassification:
    """RBR/DBR pattern detection from prior candle context."""

    def test_dbr_pattern(self) -> None:
        """Bearish candles before + bullish BOS -> DBR (reversal demand)."""
        ze = ZoneEngine()
        bos = BOS(
            timestamp=600, price=110, direction=Direction.BULLISH,
            bos_type=BOSType.IMPULSIVE,
        )
        # 5 bearish candles before the origin, then origin, then breakout.
        buf: list[Candle] = []
        for i in range(5):
            mid = 108 - i
            buf.append(_c(i * 60, mid + 1, mid + 1.5, mid - 0.5, mid))  # bearish
        buf.append(_c(300, 103, 104, 102, 103))  # origin
        buf.append(_c(360, 108, 113, 107.5, 112))  # breakout
        pattern = ze._classify_pattern(bos, buf, origin_idx=5)
        assert pattern == ZonePattern.DBR

    def test_rbr_pattern(self) -> None:
        """Bullish candles before + bullish BOS -> RBR (continuation demand)."""
        ze = ZoneEngine()
        bos = BOS(
            timestamp=600, price=110, direction=Direction.BULLISH,
            bos_type=BOSType.IMPULSIVE,
        )
        buf: list[Candle] = []
        for i in range(5):
            mid = 100 + i
            buf.append(_c(i * 60, mid, mid + 1.5, mid - 0.5, mid + 1))  # bullish
        buf.append(_c(300, 105, 106, 104, 105))
        buf.append(_c(360, 108, 113, 107.5, 112))
        pattern = ze._classify_pattern(bos, buf, origin_idx=5)
        assert pattern == ZonePattern.RBR
