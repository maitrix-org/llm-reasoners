

(define (problem BW-rand-4)
(:domain blocksworld-4ops)
(:objects a b c d )
(:init
(handempty)
(ontable a)
(ontable b)
(on c b)
(on d c)
(clear a)
(clear d)
)
(:goal
(and
(on a b)
(on c d)
(on d a))
)
)


