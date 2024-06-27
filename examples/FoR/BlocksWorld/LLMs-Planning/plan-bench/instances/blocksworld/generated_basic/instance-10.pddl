

(define (problem BW-rand-4)
(:domain blocksworld-4ops)
(:objects a b c d )
(:init
(handempty)
(ontable a)
(ontable b)
(on c b)
(on d a)
(clear c)
(clear d)
)
(:goal
(and
(on a d)
(on b c))
)
)


