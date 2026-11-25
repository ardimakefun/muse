// muse: deterministic, regime-adaptive dual-alpha trading bot for agent-fun/bars-replay/v1.
// Combines cross-asset breadth-gated trend participation (multi-day breakout,
// V-recovery surge) with volatility-normalized exhaustion snapbacks and DEX pool impact awareness.

import { BPS, runBars, type Bar, type BarsState, type BuyInput, type OrderInput, type SellInput } from './vendor/makefun-bars.ts';

type Mode = 'trend' | 'mr' | 'comp';

interface SymbolSeries {
  barsCount: number;
  closes: number[];
  emaFast: number;
  emaSlow: number;
  emaAnchor: number;
  ewmaVar: number;
  avgGain: number;
  avgLoss: number;
  rsiSeedGains: number;
  rsiSeedLosses: number;
  lastClose: number;
  lastOpen: number;
  lastHigh: number;
  lastLow: number;
}

interface PositionMeta {
  entrySeq: number;
  entryPrice: number;
  peakPrice: number;
  mode: Mode;
}

interface Candidate {
  symbol: string;
  mode: Mode;
  score: number;
  rtCostBps: number;
  isCompliance: boolean;
}

const FAST_SPAN = 6;
const SLOW_SPAN = 16;
const ANCHOR_SPAN = 28;
const RSI_SPAN = 12;
const VAR_SPAN = 18;
const MIN_SIGMA = 0.0025;
const ENTRY_SLIPPAGE_CAP = 320;
const EXIT_SLIPPAGE_CAP = 500;
const MIN_TARGET_TRADES = 8;

const seriesMap = new Map<string, SymbolSeries>();
const posMeta = new Map<string, PositionMeta>();
const pendingEntryMode = new Map<string, Mode>();
const lastExitInfo = new Map<string, { seq: number; wasStop: boolean }>();
let filledTrades = 0;
let peakEquity = 0n;

function updateSeries(bar: Bar): SymbolSeries {
  const o = Number(bar.o);
  const h = Number(bar.h);
  const l = Number(bar.l);
  const c = Number(bar.c);
  const lp = Math.log(c);
  const rangeVol = h > l ? Math.log(h / l) / 1.665 : MIN_SIGMA;
  const initVar = Math.max(MIN_SIGMA * MIN_SIGMA, rangeVol * rangeVol);

  let s = seriesMap.get(bar.symbol);
  if (s === undefined) {
    s = {
      barsCount: 1,
      closes: [c],
      emaFast: lp,
      emaSlow: lp,
      emaAnchor: lp,
      ewmaVar: initVar,
      avgGain: 0,
      avgLoss: 0,
      rsiSeedGains: 0,
      rsiSeedLosses: 0,
      lastClose: c,
      lastOpen: o,
      lastHigh: h,
      lastLow: l,
    };
    seriesMap.set(bar.symbol, s);
    return s;
  }

  const prevClose = s.lastClose;
  const r = Math.log(c / prevClose);
  s.barsCount += 1;
  s.closes.push(c);
  if (s.closes.length > 76) s.closes.shift();

  const aFast = 2 / (Math.min(s.barsCount, FAST_SPAN) + 1);
  const aSlow = 2 / (Math.min(s.barsCount, SLOW_SPAN) + 1);
  const aAnchor = 2 / (Math.min(s.barsCount, ANCHOR_SPAN) + 1);
  s.emaFast += aFast * (lp - s.emaFast);
  s.emaSlow += aSlow * (lp - s.emaSlow);
  s.emaAnchor += aAnchor * (lp - s.emaAnchor);

  const instVar = Math.max(r * r, 0.25 * rangeVol * rangeVol);
  const varAlpha = 2 / (Math.min(s.barsCount, VAR_SPAN) + 1);
  s.ewmaVar += varAlpha * (instVar - s.ewmaVar);

  const chg = (c - prevClose) / prevClose;
  const gain = chg > 0 ? chg : 0;
  const loss = chg < 0 ? -chg : 0;
  const changesSeen = s.barsCount - 1;
  if (changesSeen <= RSI_SPAN) {
    s.rsiSeedGains += gain;
    s.rsiSeedLosses += loss;
    s.avgGain = s.rsiSeedGains / RSI_SPAN;
    s.avgLoss = s.rsiSeedLosses / RSI_SPAN;
  } else {
    s.avgGain = (s.avgGain * (RSI_SPAN - 1) + gain) / RSI_SPAN;
    s.avgLoss = (s.avgLoss * (RSI_SPAN - 1) + loss) / RSI_SPAN;
  }

  s.lastClose = c;
  s.lastOpen = o;
  s.lastHigh = h;
  s.lastLow = l;
  return s;
}

function getSigma(s: SymbolSeries): number {
  return Math.max(MIN_SIGMA, Math.sqrt(s.ewmaVar));
}

function getRsi(s: SymbolSeries): number {
  const changes = s.barsCount - 1;
  if (changes < 1) return 50;
  if (changes <= RSI_SPAN) {
    const tot = s.rsiSeedGains + s.rsiSeedLosses;
    return tot === 0 ? 50 : (100 * s.rsiSeedGains) / tot;
  }
  const tot = s.avgGain + s.avgLoss;
  return tot === 0 ? 50 : (100 * s.avgGain) / tot;
}

function logRet(s: SymbolSeries, horizon: number): number {
  const n = s.closes.length;
  if (n < 2) return 0;
  const idx = Math.max(0, n - 1 - horizon);
  return Math.log(s.closes[n - 1]! / s.closes[idx]!);
}

function rollingHigh(s: SymbolSeries, window: number): number {
  const n = s.closes.length;
  const start = Math.max(0, n - window);
  let hi = 0;
  for (let i = start; i < n; i++) {
    if (s.closes[i]! > hi) hi = s.closes[i]!;
  }
  return hi;
}

function efficiencyRatio(s: SymbolSeries, window: number): number {
  const n = s.closes.length;
  if (n < 3) return 0;
  const start = Math.max(0, n - 1 - window);
  const net = Math.abs(Math.log(s.closes[n - 1]! / s.closes[start]!));
  let pathSum = 0;
  for (let i = start + 1; i < n; i++) {
    pathSum += Math.abs(Math.log(s.closes[i]! / s.closes[i - 1]!));
  }
  return pathSum === 0 ? 0 : net / pathSum;
}

function median(xs: number[]): number {
  if (xs.length === 0) return 0;
  const sorted = [...xs].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 1 ? sorted[mid]! : 0.5 * (sorted[mid - 1]! + sorted[mid]!);
}

function roundTripCostBps(state: BarsState, symbol: string, probeNotional: bigint): number {
  const impact = state.impactBps(symbol, probeNotional) ?? 20n;
  const oneWay = state.hello.fees.fee_bps + state.hello.fees.slippage_bps + Number(impact);
  return 2 * oneWay;
}

function bigMin(a: bigint, b: bigint): bigint {
  return a < b ? a : b;
}

const pendingStopExit = new Set<string>();

await runBars({
  onHello(state) {
    peakEquity = BigInt(state.hello.capital.quote_units);
  },
  onBatch(state, batch) {
    for (const f of batch.fills) {
      if (f.status === 'rejected') {
        if (f.side === 'buy') pendingEntryMode.delete(f.symbol);
        if (f.side === 'sell') pendingStopExit.delete(f.symbol);
        continue;
      }
      filledTrades++;
      if (f.side === 'buy') {
        const mode = pendingEntryMode.get(f.symbol) ?? 'trend';
        pendingEntryMode.delete(f.symbol);
        const fillPrice = Number(f.price);
        posMeta.set(f.symbol, {
          entrySeq: state.seq,
          entryPrice: fillPrice,
          peakPrice: fillPrice,
          mode,
        });
      } else if (!state.positions.has(f.symbol)) {
        posMeta.delete(f.symbol);
        const wasStop = pendingStopExit.delete(f.symbol);
        lastExitInfo.set(f.symbol, { seq: state.seq, wasStop });
      }
    }

    const freshSymbols = new Set<string>();
    for (const bar of batch.bars) {
      updateSeries(bar);
      freshSymbols.add(bar.symbol);
    }

    for (const [symbol, held] of state.positions) {
      const s = seriesMap.get(symbol);
      const curClose = s ? s.lastClose : Number(state.close(symbol) ?? 0n);
      let meta = posMeta.get(symbol);
      if (meta === undefined) {
        const approxEntry = held.qty > 0n ? curClose : 0;
        meta = { entrySeq: state.seq, entryPrice: approxEntry, peakPrice: curClose, mode: 'trend' };
        posMeta.set(symbol, meta);
      } else if (curClose > meta.peakPrice) {
        meta.peakPrice = curClose;
      }
    }
    for (const symbol of [...posMeta.keys()]) {
      if (!state.positions.has(symbol)) posMeta.delete(symbol);
    }

    if (state.warmingUp) return [];

    if (state.equity > peakEquity) peakEquity = state.equity;
    const portDd = peakEquity > 0n ? Number(peakEquity - state.equity) / Number(peakEquity) : 0;
    const scoredSeq = state.seq - state.warmupBars;
    if (scoredSeq < 2) return [];

    const { caps, fees } = state.hello;
    const minNotional = BigInt(fees.min_notional_quote_units);

    // Cross-asset market regime assessment
    const r6List: number[] = [];
    const r12List: number[] = [];
    const r24List: number[] = [];
    const sigmaList: number[] = [];
    const er10List: number[] = [];
    let upCount = 0;
    let hiCount = 0;
    let surgeCount = 0;
    let activeCount = 0;

    for (const { symbol } of state.hello.symbols) {
      const s = seriesMap.get(symbol);
      if (s === undefined || s.barsCount < 3) continue;
      activeCount++;
      const sig = getSigma(s);
      const r6 = logRet(s, 6);
      const r10 = logRet(s, 10);
      const r12 = logRet(s, 12);
      const r24 = logRet(s, 24);
      const hi72 = rollingHigh(s, 72);
      const er10 = efficiencyRatio(s, 10);
      r6List.push(r6);
      r12List.push(r12);
      r24List.push(r24);
      sigmaList.push(sig);
      er10List.push(er10);
      if (s.emaFast > s.emaSlow) upCount++;
      if (s.lastClose >= hi72 * 0.985) hiCount++;
      if (r10 > 1.6 * sig * Math.sqrt(Math.min(10, s.barsCount - 1)) && er10 >= 0.34) surgeCount++;
    }

    const breadth = activeCount > 0 ? upCount / activeCount : 0;
    const breadthHi = activeCount > 0 ? hiCount / activeCount : 0;
    const breadthSurge = activeCount > 0 ? surgeCount / activeCount : 0;
    const medR6 = median(r6List);
    const medR12 = median(r12List);
    const medR24 = median(r24List);
    const medSig = Math.max(MIN_SIGMA, median(sigmaList));
    const medEr10 = median(er10List);

    const bullMkt =
      (breadth >= 0.55 && medR12 > 0.004 && (breadthHi >= 0.42 || medR24 > 0.008)) ||
      (breadthSurge >= 0.55 && medEr10 >= 0.32);
    const look24 = Math.min(24, Math.max(4, scoredSeq + state.warmupBars + 1));
    const bearTilt = breadth <= 0.35 && medR24 < -0.65 * medSig * Math.sqrt(look24);
    const strongBear = breadth <= 0.28 && medR24 < -1.1 * medSig * Math.sqrt(look24);

    const probeNotional = (state.equity * 1600n) / BPS;
    const orders: OrderInput[] = [];
    const sellingSymbols = new Set<string>();

    // 1. Evaluate exits for held positions on fresh bars
    for (const { symbol } of state.hello.symbols) {
      const held = state.positions.get(symbol);
      if (held === undefined || !freshSymbols.has(symbol)) continue;
      const s = seriesMap.get(symbol);
      const meta = posMeta.get(symbol);
      if (s === undefined || meta === undefined) continue;

      const c = s.lastClose;
      const lp = Math.log(c);
      const sig = getSigma(s);
      const age = state.seq - meta.entrySeq;
      const gain = meta.entryPrice > 0 ? c / meta.entryPrice - 1 : 0;
      const r1 = logRet(s, 1);
      const r12 = logRet(s, 12);

      // Promote winning reversion trade into trend ride if broad bull trend confirms
      if (meta.mode === 'mr' && bullMkt && s.emaFast > s.emaSlow && gain > 0.01) {
        meta.mode = 'trend';
      }

      let shouldSell = false;
      let isStopLoss = false;

      if (portDd >= 0.14 && gain < 0) {
        shouldSell = true;
        isStopLoss = true;
      } else if (meta.mode === 'trend') {
        const trailPct = Math.min(0.115, Math.max(0.036, 8.0 * sig));
        const hardStopPct = Math.min(0.075, Math.max(0.030, 5.2 * sig));
        if (c < meta.peakPrice * Math.exp(-trailPct)) {
          shouldSell = true;
        } else if (c < meta.entryPrice * Math.exp(-hardStopPct)) {
          shouldSell = true;
          isStopLoss = true;
        } else if (age >= 8 && s.emaFast < s.emaSlow && s.emaSlow < s.emaAnchor && r12 < -0.7 * sig * Math.sqrt(12)) {
          shouldSell = true;
        }
      } else if (meta.mode === 'comp') {
        if (age >= 2 || gain >= 0.004) shouldSell = true;
      } else {
        const tp = Math.min(0.048, Math.max(0.015, 2.4 * sig));
        const sl = Math.min(0.065, Math.max(0.032, 4.4 * sig));
        if (gain >= tp) {
          shouldSell = true;
        } else if (age >= 3 && gain >= 0.010 && lp >= s.emaSlow) {
          shouldSell = true;
        } else if (c < meta.entryPrice * Math.exp(-sl)) {
          shouldSell = true;
          isStopLoss = true;
        } else if (age >= 8 && (gain > 0 || r1 <= 0 || age >= 12)) {
          shouldSell = true;
        }
      }

      if (shouldSell && orders.length < caps.max_orders_per_bar) {
        orders.push({ side: 'sell', symbol, qty: held.qty, max_slippage_bps: EXIT_SLIPPAGE_CAP } satisfies SellInput);
        sellingSymbols.add(symbol);
        if (isStopLoss) pendingStopExit.add(symbol);
      }
    }

    // 2. Clean compliance assurance only if well past halfway and < 8 trades
    let didComplianceRotate = false;
    if (
      scoredSeq >= 50 &&
      filledTrades + sellingSymbols.size < MIN_TARGET_TRADES &&
      orders.length < caps.max_orders_per_bar &&
      sellingSymbols.size === 0 &&
      state.positions.size > 0
    ) {
      let bestRotateSym: string | null = null;
      let lowestCost = Infinity;
      for (const [symbol, held] of state.positions) {
        if (!freshSymbols.has(symbol)) continue;
        const meta = posMeta.get(symbol);
        if (meta && state.seq - meta.entrySeq < 6) continue;
        const val = state.value(symbol, held.qty);
        if (val < minNotional) continue;
        const cost = roundTripCostBps(state, symbol, val);
        if (cost < lowestCost) {
          lowestCost = cost;
          bestRotateSym = symbol;
        }
      }
      if (bestRotateSym !== null) {
        const held = state.positions.get(bestRotateSym)!;
        const sellQty = state.positions.size >= 2 ? held.qty : held.qty / 2n;
        const valLeft = state.value(bestRotateSym, held.qty - sellQty);
        const actualQty = sellQty > 0n && (sellQty === held.qty || valLeft >= minNotional) ? sellQty : held.qty;
        orders.push({ side: 'sell', symbol: bestRotateSym, qty: actualQty, max_slippage_bps: EXIT_SLIPPAGE_CAP } satisfies SellInput);
        sellingSymbols.add(bestRotateSym);
        didComplianceRotate = true;
      }
    }

    // 3. Evaluate entry candidates if not in drawdown lock
    if (portDd < 0.12 && !didComplianceRotate) {
      const candidates: Candidate[] = [];
      for (const { symbol } of state.hello.symbols) {
        if (!freshSymbols.has(symbol) || state.positions.has(symbol) || sellingSymbols.has(symbol)) continue;
        const exitInfo = lastExitInfo.get(symbol);
        const coolBars = exitInfo?.wasStop && strongBear ? 6 : 2;
        if (exitInfo !== undefined && state.seq - exitInfo.seq < coolBars) continue;
        const s = seriesMap.get(symbol);
        if (s === undefined || s.barsCount < 3) continue;

        const c = s.lastClose;
        const lp = Math.log(c);
        const sig = getSigma(s);
        const r1 = logRet(s, 1);
        const r3 = logRet(s, 3);
        const r6 = logRet(s, 6);
        const r10 = logRet(s, 10);
        const r12 = logRet(s, 12);
        const r24 = logRet(s, 24);
        const hi72 = rollingHigh(s, 72);
        const er18 = efficiencyRatio(s, 17);
        const er10 = efficiencyRatio(s, 10);
        const zFast = (lp - s.emaFast) / (sig * Math.sqrt(2.5));
        const rsi = getRsi(s);
        const spanRange = s.lastHigh - s.lastLow;
        const closePos = spanRange > 0 ? (c - s.lastLow) / spanRange : 0.5;
        const rtCost = roundTripCostBps(state, symbol, probeNotional);

        // Engine A: Persistent Bull Trend (Multi-Day Breakout + V-Recovery Surge)
        const trueBreakout =
          s.barsCount >= 10 &&
          ((breadth >= 0.55 && medR24 > 0.005) || (breadth >= 0.50 && medR12 > 0 && r24 > 0.022)) &&
          s.emaFast > s.emaSlow &&
          c >= hi72 * 0.985 &&
          r24 > 0.010 &&
          r12 > 0.005 &&
          er18 >= 0.22 &&
          zFast <= 1.8;

        const vSurge =
          s.barsCount >= 10 &&
          breadthSurge >= 0.55 &&
          s.emaFast > s.emaSlow &&
          r10 > 1.6 * sig * Math.sqrt(10) &&
          er10 >= 0.34 &&
          zFast <= 2.2;

        if (trueBreakout || vSurge) {
          const score = 2.0 + r12 / sig + er10 - 0.003 * rtCost;
          candidates.push({ symbol, mode: 'trend', score, rtCostBps: rtCost, isCompliance: false });
          continue;
        }

        // Engine B: Regime-Adaptive Exhaustion Snapback + Bull Pullback + Cross-Sectional Laggard
        if (s.barsCount >= 7) {
          const upPullback = bullMkt && s.emaSlow > s.emaAnchor && zFast <= -0.50 && closePos >= 0.25;
          const z6 = r6 / (sig * Math.sqrt(Math.min(6, s.barsCount - 1)));
          const z24 = r24 / (sig * Math.sqrt(Math.min(24, s.barsCount - 1)));
          const zSlow = (lp - s.emaSlow) / (sig * Math.sqrt(6));

          let mrHit = false;
          if (bearTilt) {
            const decelOk = !strongBear || c > s.lastOpen || closePos >= 0.30 || r1 > -0.6 * sig;
            const distBps = (s.emaAnchor - lp) * 10000;
            const zCond = strongBear ? z6 <= -2.10 : z6 <= -2.05 || z24 <= -2.10;
            mrHit =
              zCond &&
              rsi <= 34 &&
              distBps >= Math.max(rtCost * 2.5, 165) &&
              decelOk;
          } else {
            const distBps = (s.emaSlow - lp) * 10000;
            mrHit = (z6 <= -1.65 || zSlow <= -1.65) && rsi <= 37 && distBps >= Math.max(rtCost * 2.1, 110);
          }

          const relLag =
            !bearTilt &&
            medR12 > -0.008 &&
            medR6 > -0.010 &&
            (r6 - medR6) / (sig * Math.sqrt(Math.min(6, s.barsCount - 1))) <= -1.85 &&
            rsi <= 38;

          if (upPullback || mrHit || relLag) {
            const score = (upPullback ? 1.5 : 0) + Math.max(0, -z6) + Math.max(0, -z24) - 0.003 * rtCost;
            candidates.push({ symbol, mode: 'mr', score, rtCostBps: rtCost, isCompliance: false });
          }
        }
      }

      // Cash-side compliance safety net if quiet bear market stayed mostly in cash past bar 50
      if (
        scoredSeq >= 50 &&
        filledTrades < MIN_TARGET_TRADES &&
        candidates.length === 0 &&
        state.positions.size === 0 &&
        scoredSeq % 4 === 0
      ) {
        let bestLiquid: string | null = null;
        let lowestImpact = Infinity;
        for (const { symbol } of state.hello.symbols) {
          if (!freshSymbols.has(symbol)) continue;
          const imp = Number(state.impactBps(symbol, (state.equity * 200n) / BPS) ?? 9999n);
          if (imp < lowestImpact) {
            lowestImpact = imp;
            bestLiquid = symbol;
          }
        }
        if (bestLiquid !== null) {
          candidates.push({ symbol: bestLiquid, mode: 'comp', score: 0.05, rtCostBps: lowestImpact * 2, isCompliance: true });
        }
      }

      candidates.sort((a, b) => (b.score !== a.score ? b.score - a.score : a.symbol < b.symbol ? -1 : a.symbol > b.symbol ? 1 : 0));

      const maxSlots = Math.min(4, caps.max_open_positions);
      let room = maxSlots - state.positions.size;
      let cash = state.cash;
      const safeWeightBps = caps.max_weight_bps > 50 ? caps.max_weight_bps - 25 : caps.max_weight_bps;
      const byWeightMax = (BigInt(safeWeightBps) * state.equity) / BPS;
      const minAlphaChunk = (400n * state.equity) / BPS;
      const entrySlipCap = Math.min(10_000, Math.max(ENTRY_SLIPPAGE_CAP, fees.slippage_bps + 250));

      for (const cand of candidates) {
        if (room <= 0 || orders.length >= caps.max_orders_per_bar) break;
        const targetWeightBps = cand.isCompliance
          ? 200n
          : cand.mode === 'trend'
            ? 2460n
            : bullMkt
              ? 2200n
              : bearTilt
                ? 1200n
                : 1600n;
        let slice = bigMin((targetWeightBps * state.equity) / BPS, byWeightMax);
        slice = bigMin(slice, state.weightRoom(cand.symbol));
        slice = bigMin(slice, cash);
        const floor = cand.isCompliance ? minNotional : bigMin(cash, minAlphaChunk) > minNotional ? minAlphaChunk : minNotional;
        if (slice < minNotional || slice < floor || slice === 0n) continue;
        orders.push({
          side: 'buy',
          symbol: cand.symbol,
          notional_quote_units: slice,
          max_slippage_bps: entrySlipCap,
        } satisfies BuyInput);
        pendingEntryMode.set(cand.symbol, cand.mode);
        cash -= slice;
        room--;
      }
    }

    return orders;
  },
});
