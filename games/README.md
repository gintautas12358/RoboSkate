## Download
### Do not commit the contents of this folder to the repository

The content of this folder can be downloaded from: 

* Linux OS: https://syncandshare.lrz.de/getlink/fiLNMD3bTMFWpDMuTZtn5sqM/linux 
* Windows OS: https://syncandshare.lrz.de/getlink/fiEBjGVGGin9YNsWuFAQfDgY/windows
* Max OS: https://syncandshare.lrz.de/getlink/fi4sFfFXUW25nvpcWwwCduQK/mac

pass: RoboSkate



## install on OS X

ERROR: "Mac You do not have permission to open the application"

When decompressing the .zip, the application contents didn't retain the execute bits. Add it back with 
```
$ sudo chmod -R 755 /path/to/app
```
Since the application was downloaded by a web browser the quarantine bits are set on the decompressed files. Remove that with 
```
$ sudo xattr -dr com.apple.quarantine /path/to/app
```