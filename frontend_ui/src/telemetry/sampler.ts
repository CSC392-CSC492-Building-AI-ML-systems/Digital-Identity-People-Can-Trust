/**
 * Throttled cursor sampling for telemetry.
 *
 * The dataset uses recorded mousemove events at most every 150 ms.
 * Thus, sample frontend inputs for the model at a similar rate to 
 * mimic the dataset it was trained on.
 */

export type Point = { x: number; y: number; ts: number; width: number; height: number };

export const SAMPLE_INTERVAL_MS = 150;

/**
 * Decide whether to send a new point.
 *
 * Returns null when there is no cursor position yet,
 * when less than `intervalMs` has passed since the last sent point,
 * or when the rounded position hasn't changed (idle cursor sends nothing).
 * Otherwise returns the point to send, with integer x/y and `ts = now`.
 */
export function nextSample(
  latest: { x: number; y: number } | null,
  lastSent: Point | null,
  now: number,
  viewport: { width: number; height: number },
  intervalMs: number = SAMPLE_INTERVAL_MS,
): Point | null {
    if (!latest) return null;

    const x = Math.round(latest.x);
    const y = Math.round(latest.y);

    if (lastSent) {
        if (now - lastSent.ts < intervalMs) return null;
        if (x === lastSent.x && y === lastSent.y) return null;
    }

    return { x, y, ts: now, width: viewport.width, height: viewport.height };
}
