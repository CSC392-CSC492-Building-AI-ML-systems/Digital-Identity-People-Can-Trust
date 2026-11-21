/**
 * HMAC-SHA256 signing for telemetry points.
 *
 * Must stay byte-for-byte compatible with `submit_telemetry` in app.py, which
 * verifies `hmac.new(hmac_secret.encode('utf-8'), f"x:{x}|y:{y}|ts:{timestamp}", sha256)`.
 */

const encoder = new TextEncoder();

// The exact string the backend signs. x, y and ts must already be integers
export function canonicalPayload(x: number, y: number, ts: number): string {
  return `x:${x}|y:${y}|ts:${ts}`;
}

// Import the session's hmac_secret. Key is the secret's UTF-8 bytes, not hex-decoded
export function importHmacKey(secret: string): Promise<CryptoKey> {
  return crypto.subtle.importKey(
    'raw',
    encoder.encode(secret),
    { name: 'HMAC', hash: 'SHA-256' },
    false,
    ['sign'],
  );
}

// Sign a point; return a 64-character lowercase hex
export async function signPoint(key: CryptoKey, x: number, y: number, ts: number): Promise<string> {
  const mac = await crypto.subtle.sign('HMAC', key, encoder.encode(canonicalPayload(x, y, ts)));
  return Array.from(new Uint8Array(mac), (b) => b.toString(16).padStart(2, '0')).join('');
}
