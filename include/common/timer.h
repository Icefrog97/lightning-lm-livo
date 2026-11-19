#pragma once

#ifndef LIDARSLAM_TIMER_HPP
#define LIDARSLAM_TIMER_HPP

#include <ctime>
#include <string>
#include <sys/time.h>

namespace lightning
{
    inline std::string get_time_us()
    {
        struct timeval tv;
        gettimeofday(&tv, nullptr);
        struct tm tm_time;
        localtime_r(&tv.tv_sec, &tm_time);

        char buf[20];

        int h = tm_time.tm_hour;
        int m = tm_time.tm_min;
        int s = tm_time.tm_sec;
        int us = tv.tv_usec;

        buf[0] = '0' + (h / 10);
        buf[1] = '0' + (h % 10);
        buf[2] = ':';
        buf[3] = '0' + (m / 10);
        buf[4] = '0' + (m % 10);
        buf[5] = ':';
        buf[6] = '0' + (s / 10);
        buf[7] = '0' + (s % 10);
        buf[8] = '.';

        for (int i = 14; i >= 9; --i)
        {
            buf[i] = '0' + (us % 10);
            us /= 10;
        }
        buf[15] = '\0';

        return "[" + std::string(buf) + "]";
    }
}

#endif
