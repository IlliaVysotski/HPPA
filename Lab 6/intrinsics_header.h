#pragma once

#define _CUSTOM_ONES(_NUM_)                                 \
        ((0x1 << (_NUM_)) - 1)

#define _CUSTOM_ONES_POS(_NUM_, _POS_)                      \
        (_CUSTOM_ONES(_NUM_) << (_POS_))

#define _CUSTOM_INT_ANY_GET(_NUM_, _INDEX_, _NUM_BITS_)     \
        (((_NUM_) >> ((_INDEX_) * (_NUM_BITS_))) &          \
            (_CUSTOM_ONES(_NUM_BITS_)))

#define _CUSTOM_INT_ANY_SET(_NUM_, _INDEX_, _NUM_BITS_)     \
        (((_NUM_) & (_CUSTOM_ONES(_NUM_BITS_))) <<          \
            ((_INDEX_) * (_NUM_BITS_)))

#define _CUSTOM_INT2_GET(_NUM_, _INDEX_)                    \
        _CUSTOM_INT_ANY_GET(_NUM_, _INDEX_, 16)

#define _CUSTOM_INT2_SET(_NUM_, _INDEX_)                    \
        _CUSTOM_INT_ANY_SET(_NUM_, _INDEX_, 16)

#define _CUSTOM_INTRINSIC_INT2_MUL(_NUM_, _SCALE_)			\
        (_CUSTOM_INT2_SET(((_CUSTOM_INT2_GET((_NUM_), 0))	\
			* (_SCALE_)), 0)								\
            | _CUSTOM_INT2_SET((_CUSTOM_INT2_GET((_NUM_),	\
            1)) * (_SCALE_), 1))

#define _CUSTOM_INTRINSIC_INT2_MUL_UPDATE(_TARGET_, _SCALE_)\
		_TARGET_ = _CUSTOM_INTRINSIC_INT2_MUL(_TARGET_,		\
			_SCALE_)

#define _CUSTOM_INTRINSIC_INT4_TO_INT2(_NUM_, _RULE_)       \
        __byte_perm(_NUM_, 0, _RULE_)

#define _CUSTOM_INTRINSIC_INT4_HIGH_TO_INT2(_NUM_)			\
        _CUSTOM_INTRINSIC_INT4_TO_INT2(_NUM_, 0x7372)

#define _CUSTOM_INTRINSIC_INT4_LOW_TO_INT2(_NUM_)			\
        _CUSTOM_INTRINSIC_INT4_TO_INT2(_NUM_, 0x7170)

#define _CUSTOM_INTRINSIC_INT2_MIDDLE(_LOW_, _HIGH_)		\
		__byte_perm(_LOW_, _HIGH_, 0x5432)

#define _CUSTOM_INTRINSIC_INT2_TO_INT4_HIGH(_TARGET_, _NUM_)\
        _TARGET_ = __byte_perm(_TARGET_, _NUM_, 0x6410)

#define _CUSTOM_INTRINSIC_INT2_TO_INT4_LOW(_TARGET_, _NUM_)	\
        _TARGET_ = __byte_perm(_TARGET_, _NUM_, 0x3264)

#define _CUSTOM_INTRINSIC_INT2_INCREMENT(_TARGET_, _NUM_)	\
		_TARGET_ = __vadd2(_TARGET_, _NUM_)

#define _CUSTOM_INTRINSIC_INT2_CHECK_0(_TARGET_)			\
		_TARGET_ = __vmaxs2(_TARGET_, 0x00000000)

#define _CUSTOM_INTRINSIC_INT2_CHECK_255(_TARGET_)			\
		_TARGET_ = __vmins2(_TARGET_, 0x00FF00FF)

#define _CUSTOM_INTRINSIC_INT4_SORT_RGB_COLORS(				\
			_OUT_RR_, _OUT_GG_, _OUT_BB_,					\
			_IN_RGBR_, _IN_GBRG_, _IN_BRGB_					\
		)													\
        {													\
			_OUT_RR_ = __byte_perm(_IN_RGBR_, _IN_GBRG_,	\
				0x0630);									\
			_OUT_RR_ = __byte_perm(_OUT_RR_, _IN_BRGB_,		\
				0x5210);									\
															\
			_OUT_GG_ = __byte_perm(_IN_RGBR_, _IN_GBRG_,	\
				0x0741);									\
			_OUT_GG_ = __byte_perm(_OUT_GG_, _IN_BRGB_,		\
				0x6210);									\
															\
			_OUT_BB_ = __byte_perm(_IN_RGBR_, _IN_GBRG_,	\
				0x0052);									\
			_OUT_BB_ = __byte_perm(_OUT_BB_, _IN_BRGB_,		\
				0x7410);									\
		}

#define _CUSTOM_INTRINSIC_INT4_SORT_RGB_COLORS_INVERT(		\
			_OUT_RGBR_, _OUT_GBRG_, _OUT_BRGB_,				\
			_IN_RR_, _IN_GG_, _IN_BB_						\
		)													\
        {													\
			_OUT_RGBR_ = __byte_perm(_IN_RR_, _IN_GG_,		\
				0x1040);									\
			_OUT_RGBR_ = __byte_perm(_OUT_RGBR_, _IN_BB_,	\
				0x3410);									\
															\
			_OUT_GBRG_ = __byte_perm(_IN_RR_, _IN_GG_,		\
				0x6205);									\
			_OUT_GBRG_ = __byte_perm(_OUT_GBRG_, _IN_BB_,	\
				0x3250);									\
															\
			_OUT_BRGB_ = __byte_perm(_IN_RR_, _IN_GG_,		\
				0x0730);									\
			_OUT_BRGB_ = __byte_perm(_OUT_BRGB_, _IN_BB_,	\
				0x7216);									\
		}
