

(define (problem MY-rand-4)
(:domain mystery-4ops)
(:objects a b c d )
(:init
(harmony)
(planet a)
(planet b)
(craves c b)
(craves d c)
(province a)
(province d)
)
(:goal
(and
(craves c b)
(craves d a))
)
)


