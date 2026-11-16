import { useEffect, useRef, useState } from 'react';
import { useNavigate, useSearchParams } from 'react-router';
import { QRCodeSVG } from 'qrcode.react';
import Camera from 'lucide-react/dist/esm/icons/camera.mjs';
import Check from 'lucide-react/dist/esm/icons/check.mjs';
import X from 'lucide-react/dist/esm/icons/x.mjs';
import { Button } from './components/ui/button';

type Status = 'waiting_for_photo' | 'processing' | 'scan_failed' | 'wrong_barcode' | 'unsupported' | 'underage' | 'verified';
type Session = { session_id: string; expires_in: number };

function phoneOrigin(address: string) {
  const entered = address.trim();
  const url = new URL(/^https?:\/\//i.test(entered) ? entered : `http://${entered}`);
  if (!['http:', 'https:'].includes(url.protocol) || !url.hostname ||
      ['localhost', '127.0.0.1', '::1', '[::1]', '0.0.0.0'].includes(url.hostname) ||
      url.pathname !== '/' || url.search || url.hash) {
    throw new Error('Enter the Network URL shown by Vite, such as http://192.168.1.10:5173/');
  }
  return url.origin;
}

export function AgeVerificationScreen() {
  const navigate = useNavigate();
  const localHost = ['localhost', '127.0.0.1', '[::1]'].includes(window.location.hostname);
  const [address, setAddress] = useState(() => localHost ? localStorage.getItem('safebet-phone-origin') ?? '' : window.location.origin);
  const [session, setSession] = useState<Session | null>(() => {
    const saved = sessionStorage.getItem('safebet-verification-session');
    return saved ? { session_id: saved, expires_in: 600 } : null;
  });
  const [status, setStatus] = useState<Status>('waiting_for_photo');
  const [error, setError] = useState('');
  const creatingSession = useRef(false);
  const [retryCount, setRetryCount] = useState(0);
  const networkOrigin = (() => {
    try { return phoneOrigin(address); }
    catch { return ''; }
  })();
  const scanUrl = (() => {
    if (!session || !networkOrigin) return '';
    return `${networkOrigin}/verify-age/scan?session=${encodeURIComponent(session.session_id)}`;
  })();

  useEffect(() => {
    if (session || !networkOrigin || creatingSession.current) return;
    creatingSession.current = true;
    setError('');
    void fetch('/api/verification/sessions', { method: 'POST' })
      .then(async (response) => {
        if (!response.ok) throw new Error('Could not start verification. Is python app.py running?');
        const created = await response.json() as Session;
        localStorage.setItem('safebet-phone-origin', networkOrigin);
        sessionStorage.setItem('safebet-verification-session', created.session_id);
        setSession(created);
        setStatus('waiting_for_photo');
      })
      .catch((reason) => setError(reason instanceof Error ? reason.message : 'Could not start verification.'))
      .finally(() => { creatingSession.current = false; });
  }, [session, networkOrigin, retryCount]);

  useEffect(() => {
    if (!session) return;
    let active = true;
    const check = async () => {
      try {
        const response = await fetch(`/api/verification/sessions/${encodeURIComponent(session.session_id)}`);
        if (!active) return;
        if (response.status === 404) {
          sessionStorage.removeItem('safebet-verification-session');
          setSession(null);
          setStatus('waiting_for_photo');
          setError('That QR code expired. Creating a new one…');
          return;
        }
        if (!response.ok) throw new Error('Could not check verification status.');
        const result = await response.json() as { status: Status };
        setStatus(result.status);
        setError('');
        if (result.status === 'underage') navigate('/verify-age/failure', { replace: true });
      } catch {
        if (active) setError('Cannot reach the backend. Make sure python app.py is running.');
      }
    };
    void check();
    const timer = window.setInterval(() => { void check(); }, 1500);
    return () => { active = false; window.clearInterval(timer); };
  }, [navigate, session]);

  return (
    <main className="grid min-h-svh place-items-center bg-[#221057] px-6 py-12 text-white">
      <section className="w-full max-w-2xl text-center">
        <h1 className="text-[2.25rem] font-semibold sm:text-[3rem]">{status === 'verified' ? 'Age Verification Successful' : 'Verify you are of age'}</h1>

        {localHost && (
          <div className="mx-auto mt-8 max-w-md text-left">
            <label htmlFor="network-address" className="mb-2 block text-sm font-medium">Network URL from the Vite terminal</label>
            <input id="network-address" type="url" inputMode="url" placeholder="http://192.168.1.10:5173/" value={address}
              onChange={(event) => setAddress(event.target.value)}
              className="h-11 w-full rounded-lg border border-white/40 bg-white px-3 text-slate-950" />
            <p className="mt-2 text-sm text-white/70">Use the Network URL, not localhost. Both devices need a network that allows them to connect directly; campus Wi-Fi may block this even when both show the same network name.</p>
          </div>
        )}

        {!networkOrigin && <p className="mx-auto mt-5 max-w-md text-sm text-white/70">Enter the Network URL to show the phone QR code.</p>}
        {networkOrigin && !session && !error && <p role="status" className="mt-8 text-white/85">Creating your phone QR code…</p>}
        {networkOrigin && !session && error && <Button type="button" variant="secondary" size="lg" className="mt-5 h-11 px-5" onClick={() => setRetryCount((count) => count + 1)}>Retry</Button>}

        {session && scanUrl && status !== 'verified' && (
          <div className="mx-auto mt-8 flex w-full max-w-lg flex-col items-center rounded-3xl bg-white p-6 text-center text-[#221057] sm:p-8">
            <p className="mb-5 max-w-md font-semibold">Scan this QR code with your phone to upload a photo of the back of your driver’s licence.</p>
            <div className="flex w-full justify-center">
              <QRCodeSVG value={scanUrl} size={224} level="M" includeMargin title="Open the phone scanner" className="block max-w-full" />
            </div>
            <a href={scanUrl} className="mt-2 block w-full max-w-md break-all text-center text-sm text-indigo-700 underline">{scanUrl}</a>
            <p className="mt-3 max-w-md text-sm text-slate-600">We will not store any details regarding your driver's license and are only using it to verify you are above 19. This QR code expires in 10 minutes.</p>
          </div>
        )}

        {status === 'unsupported' && <p role="status" className="mx-auto mt-5 max-w-md text-sm text-amber-200">Barcode received, but its format is not supported for age verification. You can retry with a supported driver licence.</p>}
        {status === 'processing' && <p role="status" className="mx-auto mt-5 max-w-md text-sm text-white/85">Photo received. Checking the barcode…</p>}
        {status === 'wrong_barcode' && <p role="status" className="mx-auto mt-5 max-w-md text-sm text-amber-200">We couldn’t read your driver’s licence. Please try again on your phone.</p>}
        {status === 'scan_failed' && <p role="status" className="mt-5 text-sm text-amber-200">We couldn’t read the photo. Please try again on your phone.</p>}
        {error && <p role="alert" className="mx-auto mt-5 max-w-md text-sm text-[#ffb0df]">{error}</p>}
        <div className="mt-8">
          <Button type="button" variant="secondary" size="lg" disabled={status !== 'verified'} onClick={() => navigate('/mines')} className="h-11 px-6 text-base enabled:cursor-pointer">{status === 'verified' ? 'Get Started' : 'Register'}</Button>
        </div>
      </section>
    </main>
  );
}

export function PhoneDocumentScanner() {
  const [params] = useSearchParams();
  const sessionId = params.get('session');
  const pendingKey = `safebet-upload-pending:${sessionId ?? ''}`;
  const cameraInput = useRef<HTMLInputElement>(null);
  const fileInput = useRef<HTMLInputElement>(null);
  const uploading = useRef(false);
  const [photo, setPhoto] = useState<File | null>(null);
  const [preview, setPreview] = useState('');
  const [message, setMessage] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [sessionStatus, setSessionStatus] = useState<Status | 'expired'>('waiting_for_photo');

  useEffect(() => {
    if (!sessionId) return;
    let active = true;
    const check = async () => {
      try {
        const response = await fetch(`/api/verification/sessions/${encodeURIComponent(sessionId)}`);
        if (!active) return;
        if (response.status === 404) {
          sessionStorage.removeItem(pendingKey);
          setSessionStatus('expired');
          return;
        }
        if (!response.ok) throw new Error('Could not reach the backend.');
        const result = await response.json() as { status: Status; error?: string | null };
        if (!active) return;
        if (uploading.current && result.status === 'waiting_for_photo') return;
        setSessionStatus(result.status);
        if (result.status === 'processing') {
          setMessage('Photo received. Checking the barcode…');
        } else if (result.status === 'verified') {
          sessionStorage.removeItem(pendingKey);
          setMessage('');
        } else if (result.error) {
          sessionStorage.removeItem(pendingKey);
          setMessage(result.error);
        } else if (result.status === 'waiting_for_photo' && sessionStorage.getItem(pendingKey)) {
          sessionStorage.removeItem(pendingKey);
          setMessage('The page reloaded before the photo reached the backend. Select the photo and send it again.');
        }
      } catch {
        if (active) setMessage('Could not connect to the backend. Check the computer and hotspot connection.');
      }
    };
    void check();
    const timer = window.setInterval(() => { void check(); }, 1500);
    return () => { active = false; window.clearInterval(timer); };
  }, [sessionId, pendingKey]);

  useEffect(() => {
    if (!photo) { setPreview(''); return; }
    const url = URL.createObjectURL(photo);
    setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [photo]);

  function choosePhoto(file?: File) {
    if (!file) return;
    if (file.size > 10 * 1024 * 1024) { setMessage('The photo must be 10 MB or smaller.'); return; }
    setPhoto(file);
    if (cameraInput.current) cameraInput.current.value = '';
    if (fileInput.current) fileInput.current.value = '';
    setMessage('');
  }

  async function sendPhoto() {
    if (!sessionId || !photo || submitting || sessionStatus === 'processing') return;
    uploading.current = true;
    sessionStorage.setItem(pendingKey, '1');
    setSubmitting(true);
    setMessage('Sending the photo to the backend…');
    const form = new FormData();
    form.append('document', photo);
    try {
      const response = await fetch(`/api/verification/sessions/${encodeURIComponent(sessionId)}/document`, { method: 'POST', body: form });
      const result = await response.json() as { status?: Status; error?: string };
      if (result.status) setSessionStatus(result.status);
      sessionStorage.removeItem(pendingKey);
      setMessage(response.ok ? '' : result.error ?? 'The scan failed. Please retake the photo.');
    } catch {
      setMessage('Connection interrupted. Checking whether the backend received the photo…');
    } finally { uploading.current = false; setSubmitting(false); }
  }

  return (
    <main className="min-h-svh bg-[#221057] px-5 py-10 text-white">
      <section className="mx-auto max-w-md">
        <Camera aria-hidden="true" className="size-8 text-[#ff72cf]" />
        <h1 className="mt-5 text-3xl font-semibold">Take a picture of the back of your driver’s licence</h1>
        {!sessionId || sessionStatus === 'expired' ? <p role="alert" className="mt-8 text-[#ffb0df]">This scanner link is missing or expired. Start a new scan on your computer.</p> :
          sessionStatus === 'verified' ? <p role="status" className="mt-8 flex items-center gap-2"><Check aria-hidden="true" className="size-5" />Verification complete. Return to your computer.</p> :
          sessionStatus === 'underage' ? <p role="status" className="mt-8 flex items-center gap-2"><X aria-hidden="true" className="size-5" />Age not verified.</p> : <>
            <input ref={cameraInput} type="file" accept="image/jpeg,image/png,image/webp" capture="environment" className="sr-only" onChange={(event) => choosePhoto(event.target.files?.[0])} />
            <input ref={fileInput} type="file" accept="image/jpeg,image/png,image/webp" className="sr-only" onChange={(event) => choosePhoto(event.target.files?.[0])} />
            <Button type="button" variant="secondary" size="lg" onClick={() => cameraInput.current?.click()} className="mt-8 h-12 w-full cursor-pointer gap-2 text-base"><Camera aria-hidden="true" className="size-5" />Take photo</Button>
            <button type="button" onClick={() => fileInput.current?.click()} className="mt-4 w-full cursor-pointer text-center text-sm text-[#ffb0df] underline">Choose an existing photo</button>
            {preview && !submitting && <img src={preview} alt="Photo ready to send" className="mt-6 max-h-72 w-full rounded-xl bg-white object-contain" />}
            <Button type="button" variant="secondary" size="lg" disabled={!photo || submitting || sessionStatus === 'processing'} onClick={() => { void sendPhoto(); }} className="mt-6 h-12 w-full text-base enabled:cursor-pointer">{submitting ? 'Sending…' : sessionStatus === 'processing' ? 'Checking barcode…' : 'Send photo'}</Button>
          </>}
        {message && sessionStatus !== 'verified' && <p role="status" className="mt-5 text-sm text-white/85">{message}</p>}
      </section>
    </main>
  );
}

export function AgeVerificationFailureScreen() {
  return <main className="grid min-h-svh place-items-center bg-[#221057] px-6 py-12 text-white"><section className="text-center"><h1 className="text-[2.25rem] font-semibold sm:text-[3rem]">Sorry, you are not of age</h1><p className="mt-8 text-white/85">You may now exit SafeBet.</p><div className="mx-auto mt-12 flex h-11 w-fit items-center gap-2 rounded-lg bg-[#d92d34] px-5 font-medium"><X aria-hidden="true" className="size-4" />Age not verified</div></section></main>;
}
