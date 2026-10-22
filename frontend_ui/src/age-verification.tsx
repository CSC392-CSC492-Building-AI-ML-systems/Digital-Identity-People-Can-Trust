import { useNavigate } from 'react-router';
import Check from 'lucide-react/dist/esm/icons/check.mjs';
import X from 'lucide-react/dist/esm/icons/x.mjs';
import { Button } from './components/ui/button';

export function AgeVerificationScreen({ verified = false }: { verified?: boolean }) {
  const navigate = useNavigate();

  return (
    <main className="grid min-h-svh place-items-center bg-[#221057] px-6 py-12 text-white">
      <section className="w-full max-w-2xl text-center" aria-labelledby="verification-title">
        <h1 id="verification-title" className="text-[2.25rem] leading-[1.15] font-semibold tracking-[-0.025em] sm:text-[3rem]">
          Verify you are of age
        </h1>

        <div className="mx-auto mt-8 max-w-xl space-y-4 text-base leading-relaxed text-white/85">
          <p>In compliance of Ontario law, you must be 19 years of age or older to use this site. We use Zero Knowledge Proofs (ZKPs) such that we get your age - this mechanism allows us to verify you are 19+ without getting any access to your data.</p>
          <p>Please upload a photo of the front and back of your drivers license. Again, don’t worry! We will not be storing or processing this information - your device will.</p>
        </div>

        <div className="mx-auto mt-12 flex max-w-lg flex-col items-center justify-center gap-4 sm:flex-row sm:gap-6">
          {verified ? (
            <div className="flex h-11 w-full items-center justify-center gap-2 rounded-lg border border-white/30 px-4 text-sm text-white/90 sm:w-auto">
              <span className="inline-flex size-6 shrink-0 items-center justify-center rounded-full bg-[#ff00aa] text-white ring-1 ring-white/20">
                <Check aria-hidden="true" className="size-3.5 stroke-[2.5]" />
              </span>
              19+ age verified
            </div>
          ) : (
            <Button type="button" variant="secondary" size="lg" className="h-11 w-full cursor-pointer gap-2 px-5 text-base sm:w-auto" onClick={() => navigate('/verify-age/success')}>
              <span aria-hidden="true" className="inline-flex size-6 shrink-0 items-center justify-center rounded-full bg-[#4d3776] text-xs font-semibold text-white ring-1 ring-[#221057]/15">1</span>
              Upload Images
            </Button>
          )}

          <svg aria-hidden="true" viewBox="0 0 96 20" className="hidden h-5 w-24 shrink-0 text-white/60 sm:block" fill="none">
            <path d="M2 10H91M84 3L91 10L84 17" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
          </svg>

          <Button type="button" variant="secondary" size="lg" disabled={!verified} onClick={() => navigate('/mines')} className="h-11 w-full gap-2 px-5 text-base enabled:cursor-pointer sm:w-auto">
            <span aria-hidden="true" className="inline-flex size-6 shrink-0 items-center justify-center rounded-full bg-[#4d3776] text-xs font-semibold text-white ring-1 ring-[#221057]/15">2</span>
            Register
          </Button>
        </div>
      </section>
    </main>
  );
}

export function AgeVerificationFailureScreen() {
  return (
    <main className="grid min-h-svh place-items-center bg-[#221057] px-6 py-12 text-white">
      <section className="w-full max-w-2xl text-center" aria-labelledby="verification-failure-title">
        <h1 id="verification-failure-title" className="text-[2.25rem] leading-[1.15] font-semibold tracking-[-0.025em] sm:text-[3rem]">
          Sorry, you are not of age
        </h1>

        <p className="mx-auto mt-8 max-w-xl text-base leading-relaxed text-white/85">
          Sorry, your ID has revealed that you are not of age to use SafeBet. You may now exit the website.
        </p>

        <div className="mx-auto mt-12 flex h-11 w-fit items-center justify-center gap-2 rounded-lg bg-[#d92d34] px-5 text-base font-medium text-white">
          <span className="inline-flex size-6 items-center justify-center rounded-full bg-white/20">
            <X aria-hidden="true" className="size-3.5 stroke-[2.5]" />
          </span>
          Age not verified
        </div>
      </section>
    </main>
  );
}
