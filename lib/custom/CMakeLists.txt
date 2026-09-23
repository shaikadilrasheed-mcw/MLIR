add_mlir_dialect_library(MLIRcustom
		customDialect.cpp
		customOps.cpp
		convertCustomToLoops.cpp
		convertCustomToTosaPass.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/custom

        DEPENDS
        MLIRcustomOpsIncGen
		MLIRcustomPassesIncGen

	LINK_LIBS PUBLIC
	MLIRIR
	)
