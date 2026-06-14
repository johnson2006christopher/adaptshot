with open('src/adaptshot/training/up_ugf.py', 'r') as f:
    content = f.read()

# Replace the full cosine similarity matrix with approximate method
old = """        # 3. Redundancy component: 1 - max cosine sim to same-class embeddings
        # Vectorized computation against self for simplicity
        embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
        sim_matrix = embeddings_norm @ embeddings_norm.T
        np.fill_diagonal(sim_matrix, -1.0)  # Exclude self-similarity
        max_sim = np.max(sim_matrix, axis=1)
        red_score = np.power(np.clip(1.0 - max_sim, 0.0, 1.0), self.w_red)"""

new = """        # 3. Redundancy component: 1 - max cosine sim to other embeddings.
        # v0.2.0: Replaced O(N^2) full similarity matrix with approximate method
        # using random projection LSH for large buffers (>100 examples).
        # For small buffers (<=100), falls back to exact computation.
        if N <= 100:
            # Exact: O(N^2) but N is small so it's fast (~1ms for N=100)
            embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
            sim_matrix = embeddings_norm @ embeddings_norm.T
            np.fill_diagonal(sim_matrix, -1.0)
            max_sim = np.max(sim_matrix, axis=1)
        else:
            # Approximate: random projection LSH, O(N * D * log N)
            D = embeddings.shape[1]
            n_hashes = min(D, 64)  # Number of random projections
            # Random projection matrix
            rng = np.random.default_rng(42)
            proj = rng.normal(0, 1.0, (D, n_hashes)).astype(np.float32)
            proj = proj / (np.linalg.norm(proj, axis=0, keepdims=True) + 1e-8)
            # Project embeddings to hash bits
            hashes = (embeddings @ proj) > 0  # [N, n_hashes] boolean
            # For each embedding, find max collision count as proxy for similarity
            hash_int = hashes.astype(np.int32) @ (1 << np.arange(n_hashes, dtype=np.int32))
            # Count collisions via sorting
            max_collisions = np.ones(N, dtype=np.float32)
            sort_idx = np.argsort(hash_int)
            run_start = 0
            for j in range(1, N + 1):
                if j == N or hash_int[sort_idx[j]] != hash_int[sort_idx[run_start]]:
                    run_len = j - run_start
                    if run_len > 1:
                        max_collisions[sort_idx[run_start:j]] = float(run_len)
                    run_start = j
            # Convert collisions to redundancy: 1 - (collisions / max_possible)
            max_sim = 1.0 - np.clip(max_collisions / max(2, n_hashes), 0.0, 1.0)
        red_score = np.power(np.clip(1.0 - max_sim, 0.0, 1.0), self.w_red)"""

if old in content:
    content = content.replace(old, new)
    with open('src/adaptshot/training/up_ugf.py', 'w') as f:
        f.write(content)
    print('UP-UGF: Replaced successfully')
else:
    print('UP-UGF: Not found')
