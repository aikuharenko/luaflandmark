package = "flandmark"
version = "1.0-0"

source = {
   url = "git://github.com/torch/sys"
}

description = {
   summary = "lua interface for flandmark feature points detection",
   detailed = [[
	see http://cmp.felk.cvut.cz/~uricamic/flandmark/
   ]],
   homepage = "https://github.com/torch/sys",
   license = "BSD"
}

dependencies = {
   "torch >= 7.0",
}

build = {
   type = "command",
   build_command = [[
cmake -E make_directory build;
cd build;
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$(LUA_BINDIR)/.." -DCMAKE_INSTALL_PREFIX="$(PREFIX)"; 
$(MAKE)
   ]],
   install_command = "cd build && $(MAKE) install"
}