

(define (problem MY-rand-5)
(:domain mystery-4ops)
(:objects a b c d e )
(:init
(harmony)
(planet a)
(planet b)
(planet c)
(planet d)
(craves e c)
(province a)
(province b)
(province d)
(province e)
)
(:goal
(and
(craves a d)
(craves b a)
(craves d c))
)
)


