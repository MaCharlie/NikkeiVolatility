
import qlib
from qlib.data import D
import os

cur_dir = os.getcwd()
root_dir = os.path.dirname(cur_dir)
provider_uri = os.path.join(root_dir, "data/cn_data")
path = os.path.expanduser(provider_uri)
qlib.init(provider_uri=path, region="cn")

df = D.features(['sh000300'], ['$close'], start_time='2019-01-01', end_time='2019-01-10')
print(df)