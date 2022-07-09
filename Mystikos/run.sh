myst-appbuilder Dockerfile
myst mkcpio appdir rootfs
myst exec-sgx --memory-size 4096m rootfs /usr/local/bin/python3 /app/pyTorchSplit_.py
