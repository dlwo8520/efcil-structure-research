# Structure prototype oracle audit — combined

| protocol | dataset | coverage | tasks | relative L2 | Oracle Mean Δpp | Oracle Cov Δpp | Full Oracle Δpp | new-only gap pp |
|---|---|---|---|---:|---:|---:|---:|---:|
| CAR10 | car196 | complete | `[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]` | 0.1271 | 0.5348 | 0.0746 | 1.1068 | 14.8909 |
| CAR5 | car196 | complete | `[1, 2, 3, 4, 5]` | 0.1064 | 0.3109 | 0.0249 | 0.6716 | 10.5263 |
| IA10 | imageneta | complete | `[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]` | 0.1086 | -0.0658 | 0.1317 | -0.0658 | 17.7419 |
| IA5 | imageneta | complete | `[1, 2, 3, 4, 5]` | 0.0987 | -0.3950 | -0.0658 | -0.3292 | 16.9742 |

## Cross-protocol 상태

- IA5 relative L2=0.0987, IA10=0.1086
- CAR5 relative L2=0.1064, CAR10=0.1271
- 위 표의 A–D delta는 diagnostic head 결과이며 EF-CIL main accuracy와 분리해서 해석한다.
