/** Evenly distribute n points on a sphere of radius r (golden-angle spiral). Index order = placement order,
 * so nodes sorted by community form contiguous patches on the surface. */
export function fibonacciSphere(n: number, r: number): { x: number; y: number; z: number }[] {
  const pts = [];
  const phi = Math.PI * (3 - Math.sqrt(5));
  for (let i = 0; i < n; i++) {
    const y = n === 1 ? 0 : 1 - (2 * (i + 0.5)) / n;
    const rad = Math.sqrt(1 - y * y);
    const theta = phi * i;
    pts.push({ x: Math.cos(theta) * rad * r, y: y * r, z: Math.sin(theta) * rad * r });
  }
  return pts;
}
