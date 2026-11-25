| attention | params | KV bytes/token | KV at 32k tokens | vs MHA | val bits/byte |
|---|---|---|---|---|---|
| mha | 10.82M | 9,216 | 288.0 MiB | 1.000× | 1.253 |
| gqa2 | 9.64M | 3,072 | 96.0 MiB | 0.333× | 1.254 |
| mqa | 9.34M | 1,536 | 48.0 MiB | 0.167× | 1.271 |
| mla | 10.01M | 1,536 | 48.0 MiB | 0.167× | 1.338 |
