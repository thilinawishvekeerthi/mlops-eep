PACKAGE_PATH=src/eep 
sphinx-apidoc -f -o $PACKAGE_PATH/docs/source $PACKAGE_PATH         
sphinx-build -M html $PACKAGE_PATH/docs/source $PACKAGE_PATH/docs/build
chrome $PACKAGE_PATH/docs/build/html/index.html # for linux
open -a "Google Chrome" $PACKAGE_PATH/docs/build/html/index.html # for max