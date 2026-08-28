# Bundled sample photographs

Nine real maize-leaf photographs, three per class, so that the README quickstart
runs offline with no dataset download. They are here for that purpose and for
`examples/demo/`; they are not a benchmark.

**These images are not ours.** They are from the PlantVillage dataset, pinned to
commit `7f7ecc7e1eaca78107e3affe7cb5abd9427e139a` of `spMohanty/PlantVillage-Dataset`, and are redistributed under
their own licence, which is not the licence of this repository:

- Licence: CC BY-SA 3.0 (per huggingface.co/datasets/mohanty/PlantVillage)
- Citation: Mohanty, S.P., Hughes, D.P., Salathe, M. (2016). Using deep learning for image-based plant disease detection. Frontiers in Plant Science 7:1419. https://doi.org/10.3389/fpls.2016.01419

`scripts/fetch_plantvillage.py` fetches the full class directories these came from
and verifies them against a SHA-256 manifest. The bytes here are unmodified copies
of the originals; the filenames are ours.

| bundled name | PlantVillage filename | sha256 |
|---|---|---|
| `gray_leaf_spot_1.jpg` | `00120a18-ff90-46e4-92fb-2b7a10345bd3___RS_GLSp 9357.JPG` | `22733317743ff827…` |
| `gray_leaf_spot_2.jpg` | `00a20f6f-e8bd-4453-9e25-36ea70feb626___RS_GLSp 4655.JPG` | `28afc642a85b24ee…` |
| `gray_leaf_spot_3.jpg` | `0140764c-6157-4995-9ada-9c10b81af3b8___RS_GLSp 4378.JPG` | `9c0e29744ac972c8…` |
| `healthy_maize_1.jpg` | `00031d74-076e-4aef-b040-e068cd3576eb___R.S_HL 8315 copy 2.jpg` | `08fead265cb88ae4…` |
| `healthy_maize_2.jpg` | `00665f92-adb0-41eb-bba7-9eeadecfe10e___R.S_HL 8325 copy 2.jpg` | `e248578fd042c9ef…` |
| `healthy_maize_3.jpg` | `01c3bf88-d315-42a3-8fa1-fc80a05c97f1___R.S_HL 8189 copy.jpg` | `4ef4ef9e5e805695…` |
| `northern_leaf_blight_1.jpg` | `005318c8-a5fa-4420-843b-23bdda7322c2___RS_NLB 3853 copy.jpg` | `314ccd5ea9e83c15…` |
| `northern_leaf_blight_2.jpg` | `0079c731-80f5-4fea-b6a2-4ff23a7ce139___RS_NLB 4121.JPG` | `83603f7ebab40769…` |
| `northern_leaf_blight_3.jpg` | `008d9af0-7568-4a67-bb1a-0e915836ddc0___RS_NLB 4165 copy 2.jpg` | `79b76ef168c0355b…` |
