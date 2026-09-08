add_mlir_dialect_library(MLIRcustom
		customDialect.cpp
		customOps.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/custom

        DEPENDS
        MLIRcustomOpsIncGen

	LINK_LIBS PUBLIC
	MLIRIR
	)
