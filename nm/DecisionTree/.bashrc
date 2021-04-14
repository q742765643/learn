# .bashrc

# Source global definitions
if [ -f /etc/bashrc ]; then
	. /etc/bashrc
fi

# Uncomment the following line if you don't like systemctl's auto-paging feature:
# export SYSTEMD_PAGER=

# User specific aliases and functions

#alias python='/home/learning/python/bin/python3'
#alias pip='/home/learning/python/bin/pip3'
export PATH=$PATH:/home/learning/python/bin/python3
ln -s /home/learning/python/bin/python3 /usr/local/sbin/python3
ln -s /home/learning/python/bin/pip3 /usr/local/sbin/pip3

