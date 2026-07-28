SUBDIRS := $(wildcard */Makefile)
SUBDIRS := $(SUBDIRS:%/Makefile=%)

.PHONY: clean check run $(SUBDIRS)

check: $(SUBDIRS)
	@true

run: $(SUBDIRS)
	@true

$(SUBDIRS):
	$(MAKE) -C $@ $(MAKECMDGOALS)

clean: $(SUBDIRS)
	find . -name "*~" -delete
	find . -name ".opencode*" -exec rm -rf {} +
	find . -name ".pi*" -exec rm -rf {} +
	find . -name ".claude" -type d -exec rm -rf {} +
