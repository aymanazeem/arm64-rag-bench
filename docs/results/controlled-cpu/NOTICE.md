# Diagnostic patch notices

This project's [patch](diagnostic.patch) adds untimed kernel, input and saved-state
logging to four llama.cpp files and two KleidiAI kernels. It also adds an optional
one-token state replay. These changes are project diagnostics, not upstream
functionality. Their timings are invalid for performance comparisons.

The llama.cpp portions come from revision
`8ca5a8008b57aa39f2601546c0d07912045080a5` under the
[MIT License](LICENSE-llama.cpp.txt). Copyright 2023–2026 The ggml authors.
Its KleidiAI integration file also retains Copyright 2025–2026 Arm Limited
and/or its affiliates under MIT.

The two KleidiAI kernel portions come from revision
`c80157210987eedaab2509638ef5e15950c53a8c` under
[Apache-2.0](../runtime-feasibility/LICENSE-KleidiAI.txt).
Copyright 2024–2026 Arm Limited and/or its affiliates
(`open-source-office@arm.com`).

Retain these notices and licenses with the patch. Redistributed modified source
files must also retain upstream notices and identify the changes. These licenses
cover the third-party portions; they do not license unrelated project files or
imply upstream endorsement.

[The source manifest](diagnostic-source.json) lists every affected file and its
original and modified hashes. The patch bytes match the diagnostic build.
