import { memo, useCallback, useState } from 'react';
import Bomb from 'lucide-react/dist/esm/icons/bomb.mjs';
import Gem from 'lucide-react/dist/esm/icons/gem.mjs';
import { Button } from './components/ui/button';
import { Dialog, DialogClose, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from './components/ui/dialog';
import { Input } from './components/ui/input';
import { Label } from './components/ui/label';
import { Slider } from './components/ui/slider';

type Round = {
  size: number;
  mines: number;
  bet: number;
  bombs: Set<number>;
  revealed: Set<number>;
  status: 'playing' | 'won' | 'lost' | 'cashed-out';
};

type Game = { balance: number; round: Round | null; resultOpen: boolean };

const GRID_SIZES = [25, 36, 49, 64];
const CONFETTI = Array.from({ length: 10 }, (_, index) => ({
  left: `${(index * 37 + 11) % 100}%`,
  delay: `${(index % 6) * 0.12}s`,
  color: ['#ff00aa', '#a78bfa', '#6ee7f2', '#f8b4e1'][index % 4],
}));
const money = (amount: number) => `C$${amount.toFixed(2)}`;

function multiplierFor(size: number, mines: number, diamondsFound: number) {
  let multiplier = 1;
  for (let i = 0; i < diamondsFound; i += 1) {
    multiplier *= (size - i) / (size - mines - i);
  }
  return multiplier;
}

function createBombs(size: number, count: number) {
  const cells = Array.from({ length: size }, (_, index) => index);
  for (let i = cells.length - 1; i > 0; i -= 1) {
    const random = new Uint32Array(1);
    crypto.getRandomValues(random);
    const j = random[0] % (i + 1);
    [cells[i], cells[j]] = [cells[j], cells[i]];
  }
  return new Set(cells.slice(0, count));
}

function formatMultiplier(value: number) {
  return value >= 1_000_000 ? `${value.toExponential(2)}×` : `${value.toFixed(2)}×`;
}

type TileProps = {
  index: number;
  showBomb: boolean;
  showDiamond: boolean;
  hitBomb: boolean;
  faded: boolean;
  clickable: boolean;
  onReveal: (index: number) => void;
};

const Tile = memo(function Tile({ index, showBomb, showDiamond, hitBomb, faded, clickable, onReveal }: TileProps) {
  return (
    <button type="button" disabled={!clickable} onClick={() => onReveal(index)} aria-label={`Tile ${index + 1}${showBomb ? ', bomb' : showDiamond ? ', diamond' : ''}${hitBomb ? ', hit bomb' : ''}`} className={`flex aspect-square min-w-0 items-center justify-center rounded-lg transition-colors focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-white sm:rounded-xl ${hitBomb ? 'relative z-10 bg-[#b8a8ff] ring-2 ring-inset ring-white shadow-[0_0_22px_rgba(216,198,255,0.8)]' : showBomb ? 'bg-[#292049]' : showDiamond ? 'bg-[#e60099]' : 'bg-[#6940cf] ring-1 ring-inset ring-white/10'} ${faded ? 'opacity-45' : ''} ${clickable ? 'cursor-pointer hover:bg-[#8059e2]' : ''}`}>
      {showBomb ? <Bomb aria-hidden="true" className={`size-[45%] ${hitBomb ? 'text-[#221057]' : 'text-[#d6d0ff]'}`} /> : showDiamond ? <Gem aria-hidden="true" className="size-[45%] text-white" /> : null}
    </button>
  );
});

const MinesBoard = memo(function MinesBoard({ round, gridSize, resultOpen, onReveal }: { round: Round | null; gridSize: number; resultOpen: boolean; onReveal: (index: number) => void }) {
  const boardSize = round?.size ?? gridSize;
  const roundEnded = round !== null && round.status !== 'playing';
  const showFinalBoard = roundEnded && !resultOpen;

  return (
    <section className="flex min-w-0 flex-col items-center gap-5 rounded-[32px] bg-[#3b1d92] p-4 sm:p-6" aria-labelledby="mines-title">
      <h2 id="mines-title" className="text-xl font-semibold">Mines</h2>
      <div className="grid w-full max-w-[650px] gap-1.5 rounded-[28px] bg-[#2b146d] p-3 sm:gap-2 sm:p-5" style={{ gridTemplateColumns: `repeat(${Math.sqrt(boardSize)}, minmax(0, 1fr))` }}>
        {Array.from({ length: boardSize }, (_, index) => {
          const revealed = round?.revealed.has(index) ?? false;
          const bomb = round?.bombs.has(index) ?? false;
          return (
            <Tile
              key={index}
              index={index}
              showBomb={bomb && (revealed || showFinalBoard)}
              showDiamond={!bomb && (revealed || showFinalBoard)}
              hitBomb={bomb && revealed && round?.status === 'lost'}
              faded={showFinalBoard && !revealed}
              clickable={!revealed && (round?.status === 'playing' || (roundEnded && resultOpen))}
              onReveal={onReveal}
            />
          );
        })}
      </div>
    </section>
  );
});

export function MinesGameScreen() {
  const [gridSize, setGridSize] = useState(25);
  const [mineCount, setMineCount] = useState(3);
  const [betAmount, setBetAmount] = useState('10');
  const [game, setGame] = useState<Game>({ balance: 1000, round: null, resultOpen: false });
  const [betError, setBetError] = useState('');

  const round = game.round;
  const playing = round?.status === 'playing';
  const diamondsFound = round ? [...round.revealed].filter((index) => !round.bombs.has(index)).length : 0;
  const multiplier = round ? multiplierFor(round.size, round.mines, diamondsFound) : 1;
  const availableBet = !playing && Number(betAmount) > game.balance ? game.balance.toFixed(2) : betAmount;

  function startRound() {
    const bet = Number(availableBet);
    if (!Number.isFinite(bet) || bet < 0.01 || bet > game.balance || Math.abs(Math.round(bet * 100) - bet * 100) > 1e-8) {
      setBetError(`Enter a bet from C$0.01 to ${money(game.balance)} using at most two decimal places.`);
      return;
    }
    setBetError('');
    const bombs = createBombs(gridSize, mineCount);
    setGame((current) => ({
      balance: Math.round((current.balance - bet) * 100) / 100,
      resultOpen: false,
      round: {
        size: gridSize,
        mines: mineCount,
        bet,
        bombs,
        revealed: new Set<number>(),
        status: 'playing',
      },
    }));
  }

  const revealCell = useCallback((index: number) => {
    setGame((current) => {
      const active = current.round;
      if (!active || active.status !== 'playing' || active.revealed.has(index)) return current;

      const revealed = new Set(active.revealed);
      revealed.add(index);
      const hitBomb = active.bombs.has(index);
      const won = !hitBomb && revealed.size === active.size - active.mines;
      const payout = won ? active.bet * multiplierFor(active.size, active.mines, revealed.size) : 0;
      return {
        balance: Math.round((current.balance + payout) * 100) / 100,
        round: { ...active, revealed, status: hitBomb ? 'lost' : won ? 'won' : 'playing' },
        resultOpen: hitBomb || won,
      };
    });
  }, []);

  function cashOut() {
    setGame((current) => {
      const active = current.round;
      if (!active || active.status !== 'playing' || active.revealed.size === 0) return current;

      const payout = active.bet * multiplierFor(active.size, active.mines, active.revealed.size);
      return {
        balance: Math.round((current.balance + payout) * 100) / 100,
        round: { ...active, status: 'cashed-out' },
        resultOpen: false,
      };
    });
  }

  return (
    <main className="grid min-h-svh place-items-center bg-[#221057] px-4 py-8 text-white sm:px-8 lg:px-12">
      <div className="mx-auto w-full max-w-7xl">
        <header className="mb-8 flex flex-wrap items-end justify-between gap-3">
          <h1 className="text-4xl font-semibold tracking-tight text-[#ff00aa] sm:text-5xl">Safebet</h1>
          <p className="text-sm text-white/75">Balance <strong className="text-base text-white">{money(game.balance)}</strong></p>
        </header>

        <div className="grid gap-6 lg:grid-cols-[minmax(280px,360px)_minmax(0,1fr)]">
          <section className="space-y-4 rounded-[32px] bg-[#3b1d92] p-5 sm:p-6" aria-label="Game settings">
            <div className="space-y-2 rounded-2xl bg-[#2b146d] p-4">
              <Label htmlFor="bet-amount" className="text-white">Bet Amount</Label>
              <div className="relative">
                <span className="pointer-events-none absolute inset-y-0 left-3 flex items-center text-sm text-slate-500">C$</span>
                <Input id="bet-amount" type="number" min="0.01" max={game.balance} step="0.01" inputMode="decimal" value={availableBet} onChange={(event) => {
                  const next = event.target.value;
                  setBetAmount(next !== '' && Number(next) > game.balance ? game.balance.toFixed(2) : next);
                  setBetError('');
                }} disabled={playing || game.balance < 0.01} className="h-11 bg-white pl-10 text-slate-950" />
              </div>
            </div>

            <div className="space-y-3 rounded-2xl bg-[#2b146d] p-4">
              <Label className="text-white">Grid Size</Label>
              <div className="grid grid-cols-4 gap-2">
                {GRID_SIZES.map((size) => (
                  <Button key={size} type="button" variant="secondary" size="lg" disabled={playing} aria-pressed={gridSize === size} onClick={() => { setGridSize(size); setMineCount((count) => Math.min(count, size - 1)); }} className={`h-11 px-0 text-sm ${gridSize === size ? 'bg-[#ff00aa] text-white hover:bg-[#e60099]' : ''}`}>
                    {size}
                  </Button>
                ))}
              </div>
            </div>

            <div className="space-y-4 rounded-2xl bg-[#2b146d] p-4">
              <Label htmlFor="mine-slider" className="text-white">Diamonds and Bombs</Label>
              <div className="flex items-center gap-3">
                <div className="flex min-w-10 flex-col items-center gap-1 text-[#f5baff]" aria-label={`${gridSize - mineCount} diamonds`}>
                  <Gem aria-hidden="true" className="size-5" />
                  <span className="text-sm font-semibold">{gridSize - mineCount}</span>
                </div>
                <Slider id="mine-slider" aria-label="Number of bombs" min={1} max={gridSize - 1} step={1} value={[mineCount]} onValueChange={(value) => setMineCount(Array.isArray(value) ? value[0] : value)} disabled={playing} className="flex-1 [&_[data-slot=slider-thumb]]:size-4 [&_[data-slot=slider-thumb]]:border-white [&_[data-slot=slider-thumb]]:bg-[#ff00aa] [&_[data-slot=slider-track]]:bg-white/30 [&_[data-slot=slider-range]]:bg-[#ff00aa]" />
                <div className="flex min-w-10 flex-col items-center gap-1 text-[#c9c2ff]" aria-label={`${mineCount} bombs`}>
                  <Bomb aria-hidden="true" className="size-5" />
                  <span className="text-sm font-semibold">{mineCount}</span>
                </div>
              </div>
            </div>

            {round ? (
              <div className="space-y-1 rounded-2xl bg-[#2b146d] p-4 text-sm" role="status" aria-live="polite">
                {round.status === 'lost' ? <p className="mb-2 text-base font-semibold">Bomb found — round over</p> : null}
                {round.status === 'won' ? <p className="mb-2 text-base font-semibold">All diamonds found — you win!</p> : null}
                {round.status === 'cashed-out' ? <p className="mb-2 text-base font-semibold">Cashed out</p> : null}
                <p>Diamonds found: {diamondsFound} / {round.size - round.mines}</p>
                <p>Multiplier: <strong>{formatMultiplier(multiplier)}</strong></p>
                {playing ? <p>Potential payout: {money(round.bet * multiplier)}</p> : null}
                {round.status === 'lost' ? <p className="font-semibold text-rose-200">You lost {money(round.bet)}.</p> : null}
                {round.status === 'won' ? <p className="font-semibold text-emerald-200">You won {money(round.bet * multiplier)} (profit: {money(round.bet * multiplier - round.bet)}).</p> : null}
                {round.status === 'cashed-out' ? <p className="font-semibold text-emerald-200">You received {money(round.bet * multiplier)} (profit: {money(round.bet * multiplier - round.bet)}).</p> : null}
              </div>
            ) : <p className="text-sm text-white/70">Choose your bet, grid, and bombs to start.</p>}

            {playing ? (
              <Button type="button" variant="secondary" size="lg" disabled={diamondsFound === 0} onClick={cashOut} className="h-11 w-full cursor-pointer text-base">
                {diamondsFound > 0 ? `Cash Out ${money(round.bet * multiplier)}` : 'Reveal a diamond to cash out'}
              </Button>
            ) : (
              <Button type="button" variant="secondary" size="lg" disabled={game.balance < 0.01} onClick={startRound} className="h-11 w-full cursor-pointer text-base">
                {round ? 'Play Again' : 'Place Bet'}
              </Button>
            )}
            {betError ? <p role="alert" className="text-sm text-rose-200">{betError}</p> : null}
            {game.balance < 0.01 && !playing ? (
              <Button type="button" variant="ghost" className="w-full text-white hover:bg-white/10 hover:text-white" onClick={() => { setGame({ balance: 1000, round: null, resultOpen: false }); setBetError(''); }}>
                Reset demo balance
              </Button>
            ) : null}
          </section>

          <MinesBoard round={round} gridSize={gridSize} resultOpen={game.resultOpen} onReveal={revealCell} />
        </div>
      </div>

      <Dialog open={game.resultOpen} onOpenChange={(open) => setGame((current) => ({ ...current, resultOpen: open }))}>
        <DialogContent className="overflow-hidden border-0 bg-[#faf8ff] p-7 text-[#24124f] shadow-2xl sm:max-w-sm sm:p-8" showCloseButton={false}>
          {round?.status === 'won' ? (
            <div aria-hidden="true" className="pointer-events-none absolute inset-0 overflow-hidden rounded-xl">
              {CONFETTI.map((piece, index) => (
                <span key={index} className="win-confetti-piece" style={{ left: piece.left, animationDelay: piece.delay, backgroundColor: piece.color }} />
              ))}
            </div>
          ) : null}
          <DialogHeader className="relative items-center gap-1 text-center">
            <DialogTitle className="text-2xl leading-tight font-semibold tracking-tight">
              {round?.status === 'won' ? 'You won' : 'You lost'}
            </DialogTitle>
            <DialogDescription className="sr-only">
              {round?.status === 'won' ? 'You found every diamond.' : 'You hit a bomb.'}
            </DialogDescription>
          </DialogHeader>
          {round ? (
            <div className="relative py-5 text-center">
              <p className="text-4xl font-semibold tracking-tight tabular-nums">{round.status === 'won' ? money(round.bet * multiplier) : `−${money(round.bet)}`}</p>
              {round.status === 'won' ? (
                <p className="mt-2 text-sm text-[#6b5c83]">{formatMultiplier(multiplier)} on a {money(round.bet)} bet</p>
              ) : null}
            </div>
          ) : null}
          <DialogFooter className="relative -mx-7 -mb-7 mt-0 gap-2 border-0 bg-transparent px-7 pt-0 pb-7 sm:-mx-8 sm:-mb-8 sm:px-8 sm:pb-8">
            <DialogClose render={<Button variant="outline" size="lg" className="h-10 flex-1 cursor-pointer border-[#d9d0e8] bg-transparent text-[#24124f] hover:bg-[#f0ebfa]" />}>
              View board
            </DialogClose>
            <Button type="button" size="lg" disabled={game.balance < 0.01} className="h-10 flex-1 cursor-pointer bg-[#4c2c9e] text-white hover:bg-[#3d2284]" onClick={() => {
              setGame((current) => ({ ...current, resultOpen: false }));
              startRound();
            }}>
              Play Again
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </main>
  );
}
