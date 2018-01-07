# Project Notes

### Useful Commands
* Getting the logs from the sensor: `scp -rp tree@192.168.1.240:/nsm/bro/logs/2018* data/`
    - `r` recursively
    - `p` preserve date, time and permissions
* Count the number of logged bro log events with `find . -type f | grep dns | xargs wc -`.
* Filtering out the conn-summary logs by using `find . -type f | grep conn | grep -v summary | xargs wc -l`
* Recursively uncompress gzip files by issuing `gzip -rdk 2017-12-22/dns.*`
    - `r` recursive
    - `d` is decompress
    - `k` is for keep the original
* Overwriting existing `gzip -rdf 2017-12-22/dns.*`
    - `f` don't ask permission
* Recursively removing files using `find . -type f -name "*.gz" -delete`



