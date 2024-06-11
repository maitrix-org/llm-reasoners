

(define (problem BW-rand-4)
(:domain blocksworld-4ops)
(:objects a b c d )
(:init
(handempty)
(ontable a)
(ontable b)
(ontable c)
(on d b)
(clear a)
(clear c)
(clear d)
)
(:goal
(and
(on a b)
(on b c)
(on c d))
)
)


