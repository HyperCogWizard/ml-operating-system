################################################################################
#
# Robotics Middleware Abstraction for Engineering Workbench
#
################################################################################

ROBOTICS_MIDDLEWARE_VERSION = 1.0.0
ROBOTICS_MIDDLEWARE_SITE_METHOD = local
ROBOTICS_MIDDLEWARE_SITE = $(BR2_EXTERNAL_HASSOS_PATH)/package/robotics-middleware/src
ROBOTICS_MIDDLEWARE_LICENSE = Apache License 2.0
ROBOTICS_MIDDLEWARE_LICENSE_FILES = LICENSE

define ROBOTICS_MIDDLEWARE_BUILD_CMDS
	$(TARGET_MAKE_ENV) $(MAKE) -C $(@D) CC="$(TARGET_CC)" CFLAGS="$(TARGET_CFLAGS)"
endef

define ROBOTICS_MIDDLEWARE_INSTALL_TARGET_CMDS
	$(INSTALL) -D -m 0755 $(@D)/robotics-middleware $(TARGET_DIR)/usr/bin/robotics-middleware
	$(INSTALL) -D -m 0644 $(@D)/config/robotics.conf $(TARGET_DIR)/etc/robotics/robotics.conf
	$(INSTALL) -d -m 0755 $(TARGET_DIR)/usr/lib/robotics/modules
	$(INSTALL) -D -m 0644 $(@D)/modules/*.so $(TARGET_DIR)/usr/lib/robotics/modules/
endef

define ROBOTICS_MIDDLEWARE_INSTALL_INIT_SYSTEMD
	$(INSTALL) -D -m 0644 $(@D)/contrib/robotics-middleware.service \
		$(TARGET_DIR)/usr/lib/systemd/system/robotics-middleware.service
endef

$(eval $(generic-package))