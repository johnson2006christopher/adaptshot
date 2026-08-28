# Bundled sample photographs

Twelve real maize-leaf photographs, four per class, so that the README quickstart
runs offline with no dataset download. They are here for that purpose and for
`examples/demo/`; they are not a benchmark.

`demo/` holds four more: three held-out maize leaves chosen because, against the
twelve above, they show a singleton set, a two-class set, and a wrong top-1 whose set still
contains the true class -- the progression the conference demo walks through; the
exact outcomes are sensitive to the support set and the demo assumes none of them --
and one healthy **tomato** leaf from the same dataset, as a query from a crop the
model was never shown.

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
| `gray_leaf_spot_4.jpg` | `0157de0c-5b81-44a1-abe8-eecaa6f1261a___RS_GLSp 4287.JPG` | `ea72e988e679d402…` |
| `healthy_maize_4.jpg` | `026bd735-b9f4-4eab-86f3-23df15dbec95___R.S_HL 7938 copy.jpg` | `2e13237e65d9bb1d…` |
| `northern_leaf_blight_4.jpg` | `00a14441-7a62-4034-bc40-b196aeab2785___RS_NLB 3932.JPG` | `aa69fca37d03be78…` |
| `demo/query_3_gray_leaf_spot.jpg` | `05f92471-3cd4-441b-af21-1a02304d0b6c___RS_GLSp 7315.JPG` | `bec5ce16e2d03b8b…` |
| `demo/query_2_gray_leaf_spot.jpg` | `0ce6543f-9694-4b3a-b767-9bc909f54f73___RS_GLSp 4663 copy.jpg` | `442146dc8f73c14e…` |
| `demo/query_1_healthy_maize.jpg` | `06ec8081-1520-43e3-bda7-d74467f55992___R.S_HL 7935 copy.jpg` | `f7dc3db7128eb847…` |
| `demo/query_4_tomato_healthy.jpg` | `000146ff-92a4-4db6-90ad-8fce2ae4fddd___GH_HL Leaf 259.1.JPG` | `a294299b0518a50e…` |
