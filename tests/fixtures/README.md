These fixtures contain sanitized responses observed from RunPod GraphQL on
2026-10-27. Each JSON file includes the UTC observation time and query constraints.

The read-only query selected:

- `gpuTypes { id displayName memoryInGb }` (catalogue)
- `dataCenters { id name location storageSupport gpuAvailability(input:
  $availability) { gpuTypeId available stockStatus } }`
- `gpuTypes(input: {id: "NVIDIA GeForce RTX 5090"}) { id dc0:
  lowestPrice(input: $p0) { stockStatus uninterruptablePrice } ... }`

The availability input was `gpuCount: 1, secureCloud: true, minDisk: 100`.
Each price input also contained its exact `dataCenterId`. The exact-CUDA fixture adds
`allowedCudaVersions: ["13.0"]` to both inputs. The minimum-CUDA fixture instead adds
`minCudaVersion: "13.0"`, which permits hosts with newer CUDA capability. Stock and prices were queried in
the same GraphQL operation, then split into transport-shaped fixtures for replay.
`dc0` through `dc4` correspond to the order of the data-center array.

Only the requested GPU and its five offering data centers were retained; unrelated
catalogue rows were removed. Provider false/null values and prices were preserved.
No headers, API keys, account inventory or account IDs were recorded. Tests make
no network calls. Failure cases mutate copies of these fixtures deliberately.

Schema reference: https://graphql-spec.runpod.io/ (DataCenter,
GpuAvailabilityInput, GpuLowestPriceInput, LowestPrice). Live introspection is
disabled; these fields and variable inputs were verified by authenticated queries.
`myself.datacenters` was empty on the fresh account; discovery uses the provider's
root `dataCenters` catalogue instead.
