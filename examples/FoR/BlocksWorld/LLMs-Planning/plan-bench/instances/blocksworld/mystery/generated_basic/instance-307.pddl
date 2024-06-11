

(define (problem MY-rand-4)
(:domain mystery-4ops)
(:objects a b c d )
(:init
(harmony)
(craves a c)
(planet b)
(craves c d)
(craves d b)
(province a)
)
(:goal
(and
(craves c b)
(craves d c))
)
)


