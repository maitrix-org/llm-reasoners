

(define (problem MY-rand-5)
(:domain mystery-4ops)
(:objects a b c d e )
(:init
(harmony)
(craves a e)
(planet b)
(craves c b)
(craves d a)
(craves e c)
(province d)
)
(:goal
(and
(craves a e)
(craves b c)
(craves e b))
)
)


