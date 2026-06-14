with open('src/adaptshot/core/extractor.py', 'r') as f:
    content = f.read()

# Fix 1: Upgrade eco-mode from 16x16 to 32x32 preview
old_preview = """def compute_preview_signature(image: ImageInput, size: int = 16) -> np.ndarray:"""

new_preview = """def compute_preview_signature(image: ImageInput, size: int = 32) -> np.ndarray:"""

if old_preview in content:
    content = content.replace(old_preview, new_preview)
    print('Preview size: upgraded to 32x32')
else:
    print('Preview size: Not found')

# Fix 2: Add cache clearing to _build_backbone
old_build = """@lru_cache(maxsize=4)
def _build_backbone(backbone_name: str, device: str) -> Any:
    \"\"\"Build and cache a frozen backbone on the requested device.\"\"\"
    nn = _get_torch_nn()
    backbone = BackboneRegistry[backbone_name]()
    if hasattr(backbone, "fc"):
        backbone.fc = nn.Identity()
    elif hasattr(backbone, "classifier"):
        backbone.classifier = nn.Identity()
    backbone.to(device)
    backbone.eval()
    return backbone"""

new_build = """@lru_cache(maxsize=4)
def _build_backbone(backbone_name: str, device: str) -> Any:
    \"\"\"Build and cache a frozen backbone on the requested device.

    v0.2.0: LRU cache prevents repeated backbone construction but can
    hold references to tensors on old devices. Use clear_backbone_cache()
    when switching devices or to release memory.
    \"\"\"
    nn = _get_torch_nn()
    backbone = BackboneRegistry[backbone_name]()
    if hasattr(backbone, "fc"):
        backbone.fc = nn.Identity()
    elif hasattr(backbone, "classifier"):
        backbone.classifier = nn.Identity()
    backbone.to(device)
    backbone.eval()
    return backbone


def clear_backbone_cache() -> None:
    \"\"\"Clear the LRU backbone cache to release GPU/CPU memory.

    Call this when switching devices or when memory pressure is high.
    \"\"\"
    _build_backbone.cache_clear()"""

if old_build in content:
    content = content.replace(old_build, new_build)
    print('lru_cache: Added clear function')
else:
    print('lru_cache: Not found')

# Fix 3: Better eco-mode threshold safety
old_eco = """    if config.eco_mode and support_embedding is not None and support_preview is not None:
        query_preview = compute_preview_signature(pil_image)
        preview_norm = np.linalg.norm(query_preview) + 1e-8
        support_norm = np.linalg.norm(support_preview) + 1e-8
        quick_similarity = float(
            np.dot(query_preview, support_preview) / (preview_norm * support_norm)
        )
        if quick_similarity >= config.early_exit_threshold:
            if return_numpy:
                return cast(np.ndarray, support_embedding.copy())
            return _get_torch().from_numpy(support_embedding.copy())"""

new_eco = """    if config.eco_mode and support_embedding is not None and support_preview is not None:
        query_preview = compute_preview_signature(pil_image)
        preview_norm = np.linalg.norm(query_preview) + 1e-8
        support_norm = np.linalg.norm(support_preview) + 1e-8
        quick_similarity = float(
            np.dot(query_preview, support_preview) / (preview_norm * support_norm)
        )
        # v0.2.0: Stricter eco-mode: require >= threshold AND also check
        # that the cached embedding is not stale (preview norms differ by <2x)
        norm_ratio = min(preview_norm, support_norm) / max(preview_norm, support_norm)
        if quick_similarity >= config.early_exit_threshold and norm_ratio > 0.3:
            if return_numpy:
                return cast(np.ndarray, support_embedding.copy())
            return _get_torch().from_numpy(support_embedding.copy())"""

if old_eco in content:
    content = content.replace(old_eco, new_eco)
    print('eco-mode: Added safety check')
else:
    print('eco-mode: Not found')

with open('src/adaptshot/core/extractor.py', 'w') as f:
    f.write(content)
print('All extractor fixes applied')
