import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { BrowserRouter, Link, Routes, Route, useNavigate } from 'react-router';
import CircleHelp from 'lucide-react/dist/esm/icons/circle-help.mjs';
import { Button } from './components/ui/button';
import { Input } from './components/ui/input';
import { Label } from './components/ui/label';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from './components/ui/tooltip';
import { AgeVerificationFailureScreen, AgeVerificationScreen } from './age-verification';
import { MinesGameScreen } from './mines-game';
import './styles.css';

function OpeningScreen() {
  const navigate = useNavigate();

  return (
    <main className="grid min-h-svh place-items-center bg-[#221057] px-6 py-12 text-white">
      <section className="flex w-full max-w-3xl flex-col items-center" aria-labelledby="welcome-title">
        <h1 id="welcome-title" className="text-center text-[2.25rem] leading-[1.15] font-semibold tracking-[-0.025em] sm:text-[3rem]">
          Welcome to <span className="text-[#ff00aa]">Safebet</span>
        </h1>

        <ul className="mt-10 grid max-w-full list-disc gap-4 pl-7 text-base leading-[1.5] sm:mt-12 sm:gap-5 sm:text-xl">
          <li>
            Gambling, but with safeguards to allow <u>only</u> those 19+
          </li>
          <li>Private and secure - we don’t collect your data*</li>
        </ul>

        <Button type="button" variant="secondary" size="lg" className="mt-10 h-11 cursor-pointer px-6 text-base sm:mt-12" onClick={() => navigate('/register')}>
          Get Started
        </Button>
      </section>
    </main>
  );
}

function AuthScreen({ mode }: { mode: 'register' | 'login' }) {
  const isRegistration = mode === 'register';
  const navigate = useNavigate();

  return (
    <main className="grid min-h-svh place-items-center bg-[#221057] px-6 py-12 text-white">
      <section className="w-full max-w-[380px]" aria-labelledby="auth-title">
        <h1 id="auth-title" className="text-center text-[2.25rem] leading-[1.15] font-semibold tracking-[-0.025em] sm:text-[3rem]">
          {isRegistration ? 'Register' : 'Log In'}
        </h1>

        <p className="mt-5 text-center text-base text-white/85">
          {isRegistration ? 'Already have an account? ' : 'Don’t have an account? '}
          <Link to={isRegistration ? '/login' : '/register'} className="font-medium text-[#ff72cf] underline-offset-4 hover:underline focus-visible:rounded-sm focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-white">
            {isRegistration ? 'Log in' : 'Register'}
          </Link>.
        </p>

        <div className="mt-12 space-y-5">
          <div className="space-y-2">
            <Label htmlFor="username">Username</Label>
            <Input id="username" name="username" type="text" autoComplete="username" className="h-11 border-white/35 bg-white text-slate-950" />
          </div>
          <div className="space-y-2">
            <Label htmlFor="password">Password</Label>
            <Input id="password" name="password" type="password" autoComplete={isRegistration ? 'new-password' : 'current-password'} className="h-11 border-white/35 bg-white text-slate-950" />
          </div>
        </div>

        <div className="mt-12">
          {isRegistration ? (
            <Tooltip>
              <TooltipTrigger render={<Button type="button" variant="secondary" size="lg" className="h-11 w-full cursor-pointer gap-2 px-4 text-sm sm:text-base" onClick={() => navigate('/verify-age')} />}>
                Verify Age with ZKP to Register
                <CircleHelp aria-hidden="true" className="size-4 shrink-0 opacity-70" />
              </TooltipTrigger>
              <TooltipContent className="max-w-64 text-left leading-relaxed">
                We require users to be 19 or older. Zero-knowledge proofs let us verify your age without revealing your date of birth.
              </TooltipContent>
            </Tooltip>
          ) : (
            <Button type="button" variant="secondary" size="lg" className="h-11 w-full cursor-pointer px-4 text-base">
              Log In
            </Button>
          )}
        </div>
      </section>
    </main>
  );
}

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <BrowserRouter>
      <TooltipProvider>
        <Routes>
          <Route path="/" element={<OpeningScreen />} />
          <Route path="/register" element={<AuthScreen mode="register" />} />
          <Route path="/login" element={<AuthScreen mode="login" />} />
          <Route path="/verify-age" element={<AgeVerificationScreen />} />
          <Route path="/verify-age/success" element={<AgeVerificationScreen verified />} />
          <Route path="/verify-age/failure" element={<AgeVerificationFailureScreen />} />
          <Route path="/mines" element={<MinesGameScreen />} />
        </Routes>
      </TooltipProvider>
    </BrowserRouter>
  </StrictMode>,
);
