# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.2

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/thackel/src/tensorflow/tensorflow/core/user_ops/parallelization_method_test

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/thackel/src/tensorflow/tensorflow/core/user_ops/parallelization_method_test/build

# Include any dependencies generated for this target.
include CMakeFiles/time.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/time.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/time.dir/flags.make

CMakeFiles/time.dir/time/time.cpp.o: CMakeFiles/time.dir/flags.make
CMakeFiles/time.dir/time/time.cpp.o: ../time/time.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/thackel/src/tensorflow/tensorflow/core/user_ops/parallelization_method_test/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/time.dir/time/time.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/time.dir/time/time.cpp.o -c /home/thackel/src/tensorflow/tensorflow/core/user_ops/parallelization_method_test/time/time.cpp

CMakeFiles/time.dir/time/time.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/time.dir/time/time.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/thackel/src/tensorflow/tensorflow/core/user_ops/parallelization_method_test/time/time.cpp > CMakeFiles/time.dir/time/time.cpp.i

CMakeFiles/time.dir/time/time.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/time.dir/time/time.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/thackel/src/tensorflow/tensorflow/core/user_ops/parallelization_method_test/time/time.cpp -o CMakeFiles/time.dir/time/time.cpp.s

CMakeFiles/time.dir/time/time.cpp.o.requires:
.PHONY : CMakeFiles/time.dir/time/time.cpp.o.requires

CMakeFiles/time.dir/time/time.cpp.o.provides: CMakeFiles/time.dir/time/time.cpp.o.requires
	$(MAKE) -f CMakeFiles/time.dir/build.make CMakeFiles/time.dir/time/time.cpp.o.provides.build
.PHONY : CMakeFiles/time.dir/time/time.cpp.o.provides

CMakeFiles/time.dir/time/time.cpp.o.provides.build: CMakeFiles/time.dir/time/time.cpp.o

# Object files for target time
time_OBJECTS = \
"CMakeFiles/time.dir/time/time.cpp.o"

# External object files for target time
time_EXTERNAL_OBJECTS =

libtime.a: CMakeFiles/time.dir/time/time.cpp.o
libtime.a: CMakeFiles/time.dir/build.make
libtime.a: CMakeFiles/time.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX static library libtime.a"
	$(CMAKE_COMMAND) -P CMakeFiles/time.dir/cmake_clean_target.cmake
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/time.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/time.dir/build: libtime.a
.PHONY : CMakeFiles/time.dir/build

CMakeFiles/time.dir/requires: CMakeFiles/time.dir/time/time.cpp.o.requires
.PHONY : CMakeFiles/time.dir/requires

CMakeFiles/time.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/time.dir/cmake_clean.cmake
.PHONY : CMakeFiles/time.dir/clean

CMakeFiles/time.dir/depend:
	cd /home/thackel/src/tensorflow/tensorflow/core/user_ops/parallelization_method_test/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/thackel/src/tensorflow/tensorflow/core/user_ops/parallelization_method_test /home/thackel/src/tensorflow/tensorflow/core/user_ops/parallelization_method_test /home/thackel/src/tensorflow/tensorflow/core/user_ops/parallelization_method_test/build /home/thackel/src/tensorflow/tensorflow/core/user_ops/parallelization_method_test/build /home/thackel/src/tensorflow/tensorflow/core/user_ops/parallelization_method_test/build/CMakeFiles/time.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/time.dir/depend

