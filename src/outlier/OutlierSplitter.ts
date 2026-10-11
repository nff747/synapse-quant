export class OutlierSplitter {
  /**
   * Identifies column indices whose activation peak exceeds standard threshold (e.g. 6.0).
   */
  static findOutlierColumns(activations: Float32Array, rows: number, cols: number, threshold = 6.0): number[] {
    const outliers: number[] = [];
    for (let c = 0; c < cols; c++) {
      for (let r = 0; r < rows; r++) {
        if (Math.abs(activations[r * cols + c]) > threshold) {
          outliers.push(c);
          break;
        }
      }
    }
    return outliers;
  }
}
