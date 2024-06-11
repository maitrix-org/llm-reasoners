

(define (problem MY-rand-5)
(:domain mystery-4ops)
(:objects a b c d e )
(:init
(harmony)
(craves a c)
(planet b)
(planet c)
(craves d e)
(craves e b)
(province a)
(province d)
)
(:goal
(and
(craves a d)
(craves b e)
(craves c b))
)
)


