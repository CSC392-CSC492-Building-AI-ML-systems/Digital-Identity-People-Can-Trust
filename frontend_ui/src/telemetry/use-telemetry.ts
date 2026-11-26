/**
 * React hook that streams cursor telemetry for as long as the calling page is mounted.
 *
 * It starts a session from the phone verification stored in sessionStorage,
 * samples mousemove every SAMPLE_INTERVAL_MS, sends signed points through a
 * sequential TelemetryClient, and ends the session on unmount or pagehide.
 */

import { useEffect, useState } from 'react';
import { endTelemetrySession, startTelemetrySession, TelemetryClient, type TelemetrySession } from './client';
import { nextSample, SAMPLE_INTERVAL_MS, type Point } from './sampler';

export type TelemetryStatus = 'starting' | 'active' | 'stopped' | 'unverified';
export type TelemetryState = { status: TelemetryStatus; reason: string };

// Written by age-verification.tsx when it creates the phone verification session
export const VERIFICATION_STORAGE_KEY = 'safebet-verification-session';

const START_FAILED_REASON = 'Could not start monitoring. Is python app.py running?';
const PAGE_HIDDEN_REASON = 'Session ended because the page was closed. Reload to keep playing.';

export function useTelemetry(): TelemetryState {
    const [state, setState] = useState<TelemetryState>({ status: 'starting', reason: '' });

    useEffect(() => {
        let cancelled = false;
        let session: TelemetrySession | null = null;
        let ended = false;
        let timer: number | undefined;
        let latest: { x: number; y: number } | null = null;
        let lastSent: Point | null = null;

        const onMouseMove = (event: MouseEvent) => {
            latest = { x: event.clientX, y: event.clientY };
        };

        const stopSampling = () => {
            window.removeEventListener('mousemove', onMouseMove);
            if (timer !== undefined) window.clearInterval(timer);
            timer = undefined;
        };

        const cleanup = () => {
            if (cancelled) return;
            cancelled = true;
            stopSampling();
            window.removeEventListener('pagehide', onPageHide);
            if (session && !ended) {
                ended = true;
                endTelemetrySession(session.userId);
            }
        };

        // The session is gone once the page is hidden. Lock the game so a page
        // restored from the back/forward cache can't be played unmonitored.
        function onPageHide() {
            cleanup();
            setState({ status: 'stopped', reason: PAGE_HIDDEN_REASON });
        }

        const verificationId = sessionStorage.getItem(VERIFICATION_STORAGE_KEY);
        if (!verificationId) {
            setState({ status: 'unverified', reason: '' });
            return;
        }

        window.addEventListener('pagehide', onPageHide);

        startTelemetrySession(verificationId).then(
            (started) => {
                if (cancelled) {
                    // Unmounted (e.g. StrictMode's dev double-mount) while starting: don't orphan it.
                    if (started) endTelemetrySession(started.userId);
                    return;
                }
                if (!started) {
                    setState({ status: 'unverified', reason: '' });
                    return;
                }

                session = started;
                const client = new TelemetryClient(started, (reason) => {
                    // Points still queued after cleanup get 401 once the session ends; ignore those.
                    if (cancelled) return;
                    stopSampling();
                    setState({ status: 'stopped', reason });
                });

                window.addEventListener('mousemove', onMouseMove);
                timer = window.setInterval(() => {
                    const point = nextSample(latest, lastSent, Date.now(), {
                        width: window.innerWidth,
                        height: window.innerHeight,
                    });
                    if (point) {
                        client.enqueue(point);
                        lastSent = point;
                    }
                }, SAMPLE_INTERVAL_MS);

                setState({ status: 'active', reason: '' });
            },
            () => {
                if (!cancelled) setState({ status: 'stopped', reason: START_FAILED_REASON });
            },
        );

        return cleanup;
    }, []);

    return state;
}
