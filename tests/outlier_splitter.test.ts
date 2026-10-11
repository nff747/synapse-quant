import { describe, it, expect } from 'vitest';
import { OutlierSplitter } from '../src/outlier/OutlierSplitter';

describe('OutlierSplitter', () => {
  it('detects columns exceeding threshold', () => {
    const acts = new Float32Array([
      1.0, 2.0, 15.0, 0.5,
      0.2, 8.0, 1.0, 0.1
    ]);
    const outliers = OutlierSplitter.findOutlierColumns(acts, 2, 4, 6.0);
    expect(outliers).toEqual([1, 2]);
  });
});
