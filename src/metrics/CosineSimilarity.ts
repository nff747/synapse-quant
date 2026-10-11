export class CosineSimilarity {
  static evaluate(a: ArrayLike<number>, b: ArrayLike<number>): number {
    let dot = 0.0;
    let normA = 0.0;
    let normB = 0.0;
    const len = a.length;

    for (let i = 0; i < len; i++) {
      const va = a[i];
      const vb = b[i];
      dot += va * vb;
      normA += va * va;
      normB += vb * vb;
    }

    if (normA === 0 || normB === 0) return 1.0;
    return dot / (Math.sqrt(normA) * Math.sqrt(normB));
  }
}
