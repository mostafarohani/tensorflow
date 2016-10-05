all: build_pip

build_pip: compile_pip
	bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg	

compile_pip:
	bazel build -c opt --config=cuda //tensorflow/tools/pip_package:build_pip_package

clean:
	bazel clean
