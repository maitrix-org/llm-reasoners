

(define (problem MY-rand-5)
(:domain mystery-4ops)
(:objects a b c d e )
(:init
(harmony)
(planet a)
(planet b)
(planet c)
(planet d)
(craves e b)
(province a)
(province c)
(province d)
(province e)
)
(:goal
(and
(craves b e)
(craves c d)
(craves d a)
(craves e c))
)
)


