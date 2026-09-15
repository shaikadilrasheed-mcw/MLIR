add_mlir_dialect_library(MLIRcustom
		customDialect.cpp
		customOps.cpp
		customConvertToArith.cpp
		customConvertToTosa.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/custom

        DEPENDS
        MLIRcustomOpsIncGen
		MLIRcustomPassesIncGen

	LINK_LIBS PUBLIC
	MLIRIR
	)
