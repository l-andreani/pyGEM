# -*- coding: utf-8 -*-
"""
/***************************************************************************
 pyGEM
 A stand-alone application for morpho-tectonic analyses.
                              -------------------
        begin                : 2017-03-08
        git sha              : $Format:%H$
        copyright            : (C) 2017 by Andreani Louis
        email                : andreani.louis@googlemail.com
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
"""

from datetime import datetime


def clock():
    """
    Create string with time as 'H:M:S' format.
    :return: string
    """
    time = datetime.strftime(datetime.now(), '%H:%M:%S')
    return time
