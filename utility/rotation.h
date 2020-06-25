#pragma once

#include <Open3D/Open3D.h>
#include <Eigen/Eigen>

namespace utility
{

    Eigen::Matrix3d RotMatX(double radians)
    {
        return open3d::utility::RotationMatrixX(radians);
    }

    Eigen::Matrix3d RotMatY(double radians)
    {
        return open3d::utility::RotationMatrixY(radians);
    }

    Eigen::Matrix3d RotMatZ(double radians)
    {
        return open3d::utility::RotationMatrixZ(radians);
    }

    Eigen::Matrix3d RotMatYZ(const Eigen::Vector2d &radians)
    {
        return RotMatY(radians(0)) * RotMatZ(radians(1));
    }

    Eigen::Matrix3d RotMatXYZ(const Eigen::Vector3d &radians)
    {
        return RotMatX(radians(0)) * RotMatY(radians(1)) * RotMatZ(radians(2));
    }

    // derivative of rotation matrix XYZ wrt angle x
    Eigen::Matrix3d dRotMatXYZ_ax(const Eigen::Vector3d &radians)
    {
        double c0 = cos(radians(0));
        double s0 = sin(radians(0));
        double c1 = cos(radians(1));
        double s1 = sin(radians(1));
        double c2 = cos(radians(2));
        double s2 = sin(radians(2));

        Eigen::Matrix3d dRxyz_ax;
        dRxyz_ax << 0, 0, 0,
            c0 * s1 * c2 + s0 * s2, c0 * s1 * s2 - s0 * c2, c0 * c1,
            -s0 * s1 * c2 + c0 * s2, -s0 * s1 * s2 - c0 * c2, -s0 * c1;

        return dRxyz_ax.transpose();
    }

    // derivative of rotation matrix XYZ wrt angle y
    Eigen::Matrix3d dRotMatXYZ_ay(const Eigen::Vector3d &radians)
    {
        double c0 = cos(radians(0));
        double s0 = sin(radians(0));
        double c1 = cos(radians(1));
        double s1 = sin(radians(1));
        double c2 = cos(radians(2));
        double s2 = sin(radians(2));

        Eigen::Matrix3d dRxyz_ay;
        dRxyz_ay << -c2 * s1, -s2 * s1, -c1,
            s0 * c1 * c2, s0 * s1 * s2, -s0 * s1,
            c0 * c1 * c2, c0 * c1 * s2, -c0 * s1;

        return dRxyz_ay.transpose();
    }

    // derivative of rotation matrix XYZ wrt angle z
    Eigen::Matrix3d dRotMatXYZ_az(const Eigen::Vector3d &radians)
    {
        double c0 = cos(radians(0));
        double s0 = sin(radians(0));
        double c1 = cos(radians(1));
        double s1 = sin(radians(1));
        double c2 = cos(radians(2));
        double s2 = sin(radians(2));

        Eigen::Matrix3d dRxyz_az;
        dRxyz_az << -c1 * s2, c1 * c2, 0,
            -s0 * s1 * s2 - c0 * c2, s0 * s1 * c2 - c0 * s2, 0,
            -c0 * s1 * s2 + s0 * c2, c0 * s1 * c2 + s0 * s2, 0;

        return dRxyz_az.transpose();
    }

    // derivative of rotation matrix YZ wrt angle y
    Eigen::Matrix3d dRotMatYZ_ay(const Eigen::Vector2d &radians)
    {
        double c0 = cos(radians(0));
        double s0 = sin(radians(0));
        double c1 = cos(radians(1));
        double s1 = sin(radians(1));

        Eigen::Matrix3d dRyz_ay;
        dRyz_ay << -c1 * s0, -s1 * s0, -c0,
            0, 0, 0,
            c0 * c1, c1 * s1, -s0;

        return dRyz_ay.transpose();
    }

    // derivative of rotation matrix YZ wrt angle z
    Eigen::Matrix3d dRotMatYZ_az(const Eigen::Vector2d &radians)
    {
        double c0 = cos(radians(0));
        double s0 = sin(radians(0));
        double c1 = cos(radians(1));
        double s1 = sin(radians(1));

        Eigen::Matrix3d dRyz_az;
        dRyz_az << -c0 * s1, c0 * c1, 0,
            c1, -s1, 0,
            -s0 * s1, s0 * c1, 0;

        return dRyz_az;
    }

    // derivative of rotation matrix Z wrt angle
    Eigen::Matrix3d dRotMatZ(const double radian)
    {
        double c = cos(radian);
        double s = sin(radian);

        Eigen::Matrix3d dRz;
        dRz << -s, c, 0,
            -c, -s, 0,
            0, 0, 0;

        return dRz;
    }

} // namespace utility