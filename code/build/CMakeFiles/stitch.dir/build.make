# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
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
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/ddong/image_stitch/code

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/ddong/image_stitch/code/build

# Include any dependencies generated for this target.
include CMakeFiles/stitch.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/stitch.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/stitch.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/stitch.dir/flags.make

CMakeFiles/stitch.dir/main.cpp.o: CMakeFiles/stitch.dir/flags.make
CMakeFiles/stitch.dir/main.cpp.o: ../main.cpp
CMakeFiles/stitch.dir/main.cpp.o: CMakeFiles/stitch.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ddong/image_stitch/code/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/stitch.dir/main.cpp.o"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/stitch.dir/main.cpp.o -MF CMakeFiles/stitch.dir/main.cpp.o.d -o CMakeFiles/stitch.dir/main.cpp.o -c /home/ddong/image_stitch/code/main.cpp

CMakeFiles/stitch.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/stitch.dir/main.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ddong/image_stitch/code/main.cpp > CMakeFiles/stitch.dir/main.cpp.i

CMakeFiles/stitch.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/stitch.dir/main.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ddong/image_stitch/code/main.cpp -o CMakeFiles/stitch.dir/main.cpp.s

CMakeFiles/stitch.dir/harris_self.cpp.o: CMakeFiles/stitch.dir/flags.make
CMakeFiles/stitch.dir/harris_self.cpp.o: ../harris_self.cpp
CMakeFiles/stitch.dir/harris_self.cpp.o: CMakeFiles/stitch.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ddong/image_stitch/code/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/stitch.dir/harris_self.cpp.o"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/stitch.dir/harris_self.cpp.o -MF CMakeFiles/stitch.dir/harris_self.cpp.o.d -o CMakeFiles/stitch.dir/harris_self.cpp.o -c /home/ddong/image_stitch/code/harris_self.cpp

CMakeFiles/stitch.dir/harris_self.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/stitch.dir/harris_self.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ddong/image_stitch/code/harris_self.cpp > CMakeFiles/stitch.dir/harris_self.cpp.i

CMakeFiles/stitch.dir/harris_self.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/stitch.dir/harris_self.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ddong/image_stitch/code/harris_self.cpp -o CMakeFiles/stitch.dir/harris_self.cpp.s

CMakeFiles/stitch.dir/brief_self.cpp.o: CMakeFiles/stitch.dir/flags.make
CMakeFiles/stitch.dir/brief_self.cpp.o: ../brief_self.cpp
CMakeFiles/stitch.dir/brief_self.cpp.o: CMakeFiles/stitch.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ddong/image_stitch/code/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/stitch.dir/brief_self.cpp.o"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/stitch.dir/brief_self.cpp.o -MF CMakeFiles/stitch.dir/brief_self.cpp.o.d -o CMakeFiles/stitch.dir/brief_self.cpp.o -c /home/ddong/image_stitch/code/brief_self.cpp

CMakeFiles/stitch.dir/brief_self.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/stitch.dir/brief_self.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ddong/image_stitch/code/brief_self.cpp > CMakeFiles/stitch.dir/brief_self.cpp.i

CMakeFiles/stitch.dir/brief_self.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/stitch.dir/brief_self.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ddong/image_stitch/code/brief_self.cpp -o CMakeFiles/stitch.dir/brief_self.cpp.s

CMakeFiles/stitch.dir/brute_force_match.cpp.o: CMakeFiles/stitch.dir/flags.make
CMakeFiles/stitch.dir/brute_force_match.cpp.o: ../brute_force_match.cpp
CMakeFiles/stitch.dir/brute_force_match.cpp.o: CMakeFiles/stitch.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ddong/image_stitch/code/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/stitch.dir/brute_force_match.cpp.o"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/stitch.dir/brute_force_match.cpp.o -MF CMakeFiles/stitch.dir/brute_force_match.cpp.o.d -o CMakeFiles/stitch.dir/brute_force_match.cpp.o -c /home/ddong/image_stitch/code/brute_force_match.cpp

CMakeFiles/stitch.dir/brute_force_match.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/stitch.dir/brute_force_match.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ddong/image_stitch/code/brute_force_match.cpp > CMakeFiles/stitch.dir/brute_force_match.cpp.i

CMakeFiles/stitch.dir/brute_force_match.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/stitch.dir/brute_force_match.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ddong/image_stitch/code/brute_force_match.cpp -o CMakeFiles/stitch.dir/brute_force_match.cpp.s

CMakeFiles/stitch.dir/image_stitch.cpp.o: CMakeFiles/stitch.dir/flags.make
CMakeFiles/stitch.dir/image_stitch.cpp.o: ../image_stitch.cpp
CMakeFiles/stitch.dir/image_stitch.cpp.o: CMakeFiles/stitch.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ddong/image_stitch/code/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/stitch.dir/image_stitch.cpp.o"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/stitch.dir/image_stitch.cpp.o -MF CMakeFiles/stitch.dir/image_stitch.cpp.o.d -o CMakeFiles/stitch.dir/image_stitch.cpp.o -c /home/ddong/image_stitch/code/image_stitch.cpp

CMakeFiles/stitch.dir/image_stitch.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/stitch.dir/image_stitch.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ddong/image_stitch/code/image_stitch.cpp > CMakeFiles/stitch.dir/image_stitch.cpp.i

CMakeFiles/stitch.dir/image_stitch.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/stitch.dir/image_stitch.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ddong/image_stitch/code/image_stitch.cpp -o CMakeFiles/stitch.dir/image_stitch.cpp.s

CMakeFiles/stitch.dir/ransac_homo.cpp.o: CMakeFiles/stitch.dir/flags.make
CMakeFiles/stitch.dir/ransac_homo.cpp.o: ../ransac_homo.cpp
CMakeFiles/stitch.dir/ransac_homo.cpp.o: CMakeFiles/stitch.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ddong/image_stitch/code/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object CMakeFiles/stitch.dir/ransac_homo.cpp.o"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/stitch.dir/ransac_homo.cpp.o -MF CMakeFiles/stitch.dir/ransac_homo.cpp.o.d -o CMakeFiles/stitch.dir/ransac_homo.cpp.o -c /home/ddong/image_stitch/code/ransac_homo.cpp

CMakeFiles/stitch.dir/ransac_homo.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/stitch.dir/ransac_homo.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ddong/image_stitch/code/ransac_homo.cpp > CMakeFiles/stitch.dir/ransac_homo.cpp.i

CMakeFiles/stitch.dir/ransac_homo.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/stitch.dir/ransac_homo.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ddong/image_stitch/code/ransac_homo.cpp -o CMakeFiles/stitch.dir/ransac_homo.cpp.s

CMakeFiles/stitch.dir/rm_crack.cpp.o: CMakeFiles/stitch.dir/flags.make
CMakeFiles/stitch.dir/rm_crack.cpp.o: ../rm_crack.cpp
CMakeFiles/stitch.dir/rm_crack.cpp.o: CMakeFiles/stitch.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ddong/image_stitch/code/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object CMakeFiles/stitch.dir/rm_crack.cpp.o"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/stitch.dir/rm_crack.cpp.o -MF CMakeFiles/stitch.dir/rm_crack.cpp.o.d -o CMakeFiles/stitch.dir/rm_crack.cpp.o -c /home/ddong/image_stitch/code/rm_crack.cpp

CMakeFiles/stitch.dir/rm_crack.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/stitch.dir/rm_crack.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ddong/image_stitch/code/rm_crack.cpp > CMakeFiles/stitch.dir/rm_crack.cpp.i

CMakeFiles/stitch.dir/rm_crack.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/stitch.dir/rm_crack.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ddong/image_stitch/code/rm_crack.cpp -o CMakeFiles/stitch.dir/rm_crack.cpp.s

CMakeFiles/stitch.dir/bright_consistency.cpp.o: CMakeFiles/stitch.dir/flags.make
CMakeFiles/stitch.dir/bright_consistency.cpp.o: ../bright_consistency.cpp
CMakeFiles/stitch.dir/bright_consistency.cpp.o: CMakeFiles/stitch.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ddong/image_stitch/code/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building CXX object CMakeFiles/stitch.dir/bright_consistency.cpp.o"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/stitch.dir/bright_consistency.cpp.o -MF CMakeFiles/stitch.dir/bright_consistency.cpp.o.d -o CMakeFiles/stitch.dir/bright_consistency.cpp.o -c /home/ddong/image_stitch/code/bright_consistency.cpp

CMakeFiles/stitch.dir/bright_consistency.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/stitch.dir/bright_consistency.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ddong/image_stitch/code/bright_consistency.cpp > CMakeFiles/stitch.dir/bright_consistency.cpp.i

CMakeFiles/stitch.dir/bright_consistency.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/stitch.dir/bright_consistency.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ddong/image_stitch/code/bright_consistency.cpp -o CMakeFiles/stitch.dir/bright_consistency.cpp.s

# Object files for target stitch
stitch_OBJECTS = \
"CMakeFiles/stitch.dir/main.cpp.o" \
"CMakeFiles/stitch.dir/harris_self.cpp.o" \
"CMakeFiles/stitch.dir/brief_self.cpp.o" \
"CMakeFiles/stitch.dir/brute_force_match.cpp.o" \
"CMakeFiles/stitch.dir/image_stitch.cpp.o" \
"CMakeFiles/stitch.dir/ransac_homo.cpp.o" \
"CMakeFiles/stitch.dir/rm_crack.cpp.o" \
"CMakeFiles/stitch.dir/bright_consistency.cpp.o"

# External object files for target stitch
stitch_EXTERNAL_OBJECTS =

stitch: CMakeFiles/stitch.dir/main.cpp.o
stitch: CMakeFiles/stitch.dir/harris_self.cpp.o
stitch: CMakeFiles/stitch.dir/brief_self.cpp.o
stitch: CMakeFiles/stitch.dir/brute_force_match.cpp.o
stitch: CMakeFiles/stitch.dir/image_stitch.cpp.o
stitch: CMakeFiles/stitch.dir/ransac_homo.cpp.o
stitch: CMakeFiles/stitch.dir/rm_crack.cpp.o
stitch: CMakeFiles/stitch.dir/bright_consistency.cpp.o
stitch: CMakeFiles/stitch.dir/build.make
stitch: /usr/local/lib/libopencv_gapi.so.4.7.0
stitch: /usr/local/lib/libopencv_highgui.so.4.7.0
stitch: /usr/local/lib/libopencv_ml.so.4.7.0
stitch: /usr/local/lib/libopencv_objdetect.so.4.7.0
stitch: /usr/local/lib/libopencv_photo.so.4.7.0
stitch: /usr/local/lib/libopencv_stitching.so.4.7.0
stitch: /usr/local/lib/libopencv_video.so.4.7.0
stitch: /usr/local/lib/libopencv_videoio.so.4.7.0
stitch: /usr/local/lib/libopencv_imgcodecs.so.4.7.0
stitch: /usr/local/lib/libopencv_dnn.so.4.7.0
stitch: /usr/local/lib/libopencv_calib3d.so.4.7.0
stitch: /usr/local/lib/libopencv_features2d.so.4.7.0
stitch: /usr/local/lib/libopencv_flann.so.4.7.0
stitch: /usr/local/lib/libopencv_imgproc.so.4.7.0
stitch: /usr/local/lib/libopencv_core.so.4.7.0
stitch: CMakeFiles/stitch.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/ddong/image_stitch/code/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Linking CXX executable stitch"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/stitch.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/stitch.dir/build: stitch
.PHONY : CMakeFiles/stitch.dir/build

CMakeFiles/stitch.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/stitch.dir/cmake_clean.cmake
.PHONY : CMakeFiles/stitch.dir/clean

CMakeFiles/stitch.dir/depend:
	cd /home/ddong/image_stitch/code/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ddong/image_stitch/code /home/ddong/image_stitch/code /home/ddong/image_stitch/code/build /home/ddong/image_stitch/code/build /home/ddong/image_stitch/code/build/CMakeFiles/stitch.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/stitch.dir/depend

