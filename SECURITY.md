# 보안 정책 / Security policy

보안 취약점을 발견하셨다면 **공개 이슈 대신** 이 레포의 **Security → Report a vulnerability**(비공개 취약점 신고)로 알려 주세요. 확인 후 답변드리고, 고친 버전을 릴리스하면서 함께 안내합니다.

If you find a security issue, please **do not open a public issue**. Use **Security → Report a vulnerability** in this repository to report it privately. We will reply after checking and announce the fix with a release.

semsplit 은 네트워크 포트를 열지 않습니다. CLI 는 `--embeddings ollama:<model>` 또는 `OLLAMA_HOST` 로 직접 고른 Ollama 호스트에만 접속합니다. / semsplit does not open any network port; the CLI connects only to the Ollama host you choose with `--embeddings ollama:<model>` / `OLLAMA_HOST`.
