

(define (problem MY-rand-5)
(:domain mystery-4ops)
(:objects a b c d e )
(:init
(harmony)
(planet a)
(craves b e)
(planet c)
(planet d)
(craves e c)
(province a)
(province b)
(province d)
)
(:goal
(and
(craves a e)
(craves d c)
(craves e b))
)
)


