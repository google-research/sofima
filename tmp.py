from sofima.zarr import zarr_io

output_bucket = 'sofima-test-bucket'
fused_shape_5d = [1, 1, 3543, 867, 576]
output_path = 'tmp.zarr'

ds_out = zarr_io.write_zarr(output_bucket, fused_shape_5d, output_path)