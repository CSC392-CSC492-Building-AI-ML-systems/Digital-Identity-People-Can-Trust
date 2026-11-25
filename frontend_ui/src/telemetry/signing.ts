/**
 * HMAC-SHA256 signing for telemetry points.
 *
 * Must stay byte-for-byte compatible with `submit_telemetry` in app.py, which
 * verifies `hmac.new(hmac_secret.encode('utf-8'), f"x:{x}|y:{y}|ts:{timestamp}", sha256)`.
 */

import { hmac } from '@noble/hashes/hmac.js';
import { sha256 } from '@noble/hashes/sha2.js';
import { bytesToHex } from '@noble/hashes/utils.js';

// Pure-JS HMAC rather than Web Crypto. crypto.subtle only exists on secure
// origins (HTTPS or localhost), so it's undefined when the app is opened at
// http://<LAN-IP>; e.g. from a phone during development.

const encoder = new TextEncoder();

// The exact string the backend signs. x, y and ts must already be integers
export function canonicalPayload(x: number, y: number, ts: number): string {
  return `x:${x}|y:${y}|ts:${ts}`;
}

// Import the session's hmac_secret. Key is the secret's UTF-8 bytes, not hex-decoded
export function importHmacKey(secret: string): Uint8Array {
  return encoder.encode(secret);
}

// Sign a point; return a 64-character lowercase hex
export function signPoint(key: Uint8Array, x: number, y: number, ts: number): string {
  return bytesToHex(hmac(sha256, key, encoder.encode(canonicalPayload(x, y, ts))));
}
