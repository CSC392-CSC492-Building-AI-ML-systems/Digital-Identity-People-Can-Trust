/**
 * Telemetry session API and a strictly sequential, signed send queue.
 *
 * Points are sent one request at a time so they always reach Flask in
 * timestamp order.
 * 
 * The backend stops a session on any non-increasing timestamp.
 */

import type { Point } from './sampler';
import { importHmacKey, signPoint } from './signing';

export type TelemetrySession = { userId: string; hmacSecret: string };

export const MAX_QUEUE = 50;
export const MAX_NETWORK_FAILURES = 3;

const JSON_HEADERS = { 'Content-Type': 'application/json' };

/**
 * Open a telemetry session for a completed phone verification.
 *
 * Returns null when the backend refuses it (403: missing, expired or
 * unverified). Throws on any other failure, including network errors.
 */
export async function startTelemetrySession(
    verificationSessionId: string,
    fetchFn: typeof fetch = fetch,
): Promise<TelemetrySession | null> {
    const response = await fetchFn('/api/session/start', {
        method: 'POST',
        headers: JSON_HEADERS,
        body: JSON.stringify({ verification_session_id: verificationSessionId }),
    });

    if (response.status === 403) return null;
    if (response.status !== 200) throw new Error(`Session start failed (HTTP ${response.status}).`);
    const body = await response.json();

    return { userId: body.user_id, hmacSecret: body.hmac_secret };
}

// End a session with a beacon, which still goes out while the page unloads
export function endTelemetrySession(
    userId: string,
    beacon: (url: string, data: Blob) => boolean = navigator.sendBeacon.bind(navigator),
): void {
    const data = new Blob([JSON.stringify({ user_id: userId })], { type: 'application/json' });
    beacon('/api/session/end', data);
}

export class TelemetryClient {
    private queue: Point[] = [];
    private sending = false;
    private isStopped = false;
    private failures = 0;
    private key: Promise<CryptoKey> | null = null;

    constructor(
        private readonly session: TelemetrySession,
        private readonly onStop: (reason: string) => void,
        private readonly fetchFn: typeof fetch = fetch,
    ) {}

    get stopped(): boolean {
        return this.isStopped;
    }

    // Queue a point for sending. Drops the oldest queued point when full; no-op once stopped
    enqueue(p: Point): void {
        if (this.isStopped) return;
        if (this.queue.length >= MAX_QUEUE) this.queue.shift();
        this.queue.push(p);
        void this.drain();
    }

    private stop(reason: string): void {
        if (this.isStopped) return;
        this.isStopped = true;
        this.queue = [];
        this.onStop(reason);
    }

    private async drain(): Promise<void> {
        if (this.sending) return;
        this.sending = true;
        try {
            while (!this.isStopped && this.queue.length > 0) {
                // Take the point off the queue before sending. A failed point is never
                // retried. If the request did reach Flask, resending it would look like
                // a replay (dt <= 0) and stop the session.
                await this.send(this.queue.shift()!);
            }
        } finally {
            this.sending = false;
        }
    }

    private async send(p: Point): Promise<void> {
        let response: Response;

        try {
            this.key ??= importHmacKey(this.session.hmacSecret);
            const signature = await signPoint(await this.key, p.x, p.y, p.ts);
            
            response = await this.fetchFn('/api/telemetry/submit', {
                method: 'POST',
                headers: JSON_HEADERS,
                body: JSON.stringify({
                    user_id: this.session.userId,
                    x: p.x,
                    y: p.y,
                    timestamp: p.ts,
                    width: p.width,
                    height: p.height,
                    signature,
                }),
            });
        } catch {
            this.failures += 1;
            if (this.failures >= MAX_NETWORK_FAILURES) this.stop('Lost connection to the server.');
            return;
        }

        this.failures = 0;
        if (response.status === 200) return;

        const body = await response.json().catch(() => ({}));
        const reason = typeof body?.reason === 'string' && body.reason ? body.reason : 'Session stopped.';
        this.stop(reason);
    }
}
